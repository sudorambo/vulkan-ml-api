/**
 * @file test_ml_graph.c
 * @brief ML graph CTS tests - creation, compilation, memory requirements.
 *
 * Verifies single-node graphs per operation type, multi-node chains,
 * scratch memory queries, graph destroy, and multiple external I/O.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

/* ------------------------------------------------------------------ */
/* Test helpers                                                        */
/* ------------------------------------------------------------------ */

static void make_tensor_desc(VkTensorDescriptionKHR *desc,
                             uint32_t *dims,
                             uint32_t dim_count,
                             VkFormat format,
                             VkTensorUsageFlagsKHR usage)
{
    desc->sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc->pNext = NULL;
    desc->tiling = VK_TENSOR_TILING_OPTIMAL_KHR;
    desc->format = format;
    desc->dimensionCount = dim_count;
    desc->pDimensions = dims;
    desc->pStrides = NULL;
    desc->usage = usage;
}

static void make_tensor_binding_external_input(VkMLTensorBindingKHR *b,
                                               uint32_t tensor_index,
                                               const VkTensorDescriptionKHR *desc)
{
    b->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    b->pNext = NULL;
    b->bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR;
    b->nodeIndex = 0;
    b->tensorIndex = tensor_index;
    b->pTensorDescription = desc;
}

static void make_tensor_binding_external_output(VkMLTensorBindingKHR *b,
                                                uint32_t tensor_index,
                                                const VkTensorDescriptionKHR *desc)
{
    b->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    b->pNext = NULL;
    b->bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR;
    b->nodeIndex = 0;
    b->tensorIndex = tensor_index;
    b->pTensorDescription = desc;
}

static void make_tensor_binding_internal(VkMLTensorBindingKHR *b,
                                         uint32_t node_index,
                                         uint32_t tensor_index,
                                         const VkTensorDescriptionKHR *desc)
{
    b->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    b->pNext = NULL;
    b->bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR;
    b->nodeIndex = node_index;
    b->tensorIndex = tensor_index;
    b->pTensorDescription = desc;
}

/* ------------------------------------------------------------------ */
/* Test 1: Single-node convolution graph                              */
/* ------------------------------------------------------------------ */

static int test_single_node_convolution(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 128, 56, 56};
    VkTensorDescriptionKHR in_desc;
    VkTensorDescriptionKHR out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescConvolutionKHR conv_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
        .pNext = NULL,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .kernelWidth = 3,
        .kernelHeight = 3,
        .strideX = 1,
        .strideY = 1,
        .dilationX = 1,
        .dilationY = 1,
        .paddingMode = VK_ML_PADDING_MODE_SAME_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
        .groupCount = 1,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
        .pOperationDesc = &conv_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "conv",
    };

    VkTensorDescriptionKHR ext_inputs[] = {in_desc};
    VkTensorDescriptionKHR ext_outputs[] = {out_desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 2: Single-node GEMM graph                                      */
/* ------------------------------------------------------------------ */

static int test_single_node_gemm(void)
{
    uint32_t a_dims[] = {64, 128};
    uint32_t b_dims[] = {128, 256};
    uint32_t out_dims[] = {64, 256};
    VkTensorDescriptionKHR a_desc, b_desc, out_desc;
    make_tensor_desc(&a_desc, a_dims, 2, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&b_desc, b_dims, 2, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 2, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescGemmKHR gemm_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR,
        .pNext = NULL,
        .transposeA = VK_FALSE,
        .transposeB = VK_FALSE,
        .alpha = 1.0f,
        .beta = 0.0f,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[2];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &a_desc);
    make_tensor_binding_external_input(&inputs[1], 1, &b_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_GEMM_KHR,
        .pOperationDesc = &gemm_desc,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "gemm",
    };

    VkTensorDescriptionKHR ext_inputs[] = {a_desc, b_desc};
    VkTensorDescriptionKHR ext_outputs[] = {out_desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 2,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 3: Single-node pooling graph                                   */
/* ------------------------------------------------------------------ */

static int test_single_node_pooling(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 64, 28, 28};
    VkTensorDescriptionKHR in_desc, out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescPoolingKHR pool_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 2,
        .windowHeight = 2,
        .strideX = 2,
        .strideY = 2,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
    };

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
        .pOperationDesc = &pool_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "pool",
    };

    VkTensorDescriptionKHR ext_inputs[] = {in_desc};
    VkTensorDescriptionKHR ext_outputs[] = {out_desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 4: Single-node activation graph                                */
/* ------------------------------------------------------------------ */

static int test_single_node_activation(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        .activationType = VK_ML_ACTIVATION_FUNCTION_RELU_KHR,
        .param0 = 0.0f,
        .param1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &desc);
    make_tensor_binding_external_output(&outputs[0], 0, &desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = &act_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "relu",
    };

    VkTensorDescriptionKHR ext_inputs[] = {desc};
    VkTensorDescriptionKHR ext_outputs[] = {desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 5: Single-node normalization graph                            */
/* ------------------------------------------------------------------ */

static int test_single_node_normalization(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescNormalizationKHR norm_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR,
        .pNext = NULL,
        .normType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR,
        .epsilon = 1e-5f,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &desc);
    make_tensor_binding_external_output(&outputs[0], 0, &desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR,
        .pOperationDesc = &norm_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "batchnorm",
    };

    VkTensorDescriptionKHR ext_inputs[] = {desc};
    VkTensorDescriptionKHR ext_outputs[] = {desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 6: Single-node elementwise graph                               */
/* ------------------------------------------------------------------ */

static int test_single_node_elementwise(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescElementwiseKHR elem_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR,
        .pNext = NULL,
        .opType = VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[2];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &desc);
    make_tensor_binding_external_input(&inputs[1], 1, &desc);
    make_tensor_binding_external_output(&outputs[0], 0, &desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR,
        .pOperationDesc = &elem_desc,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "add",
    };

    VkTensorDescriptionKHR ext_inputs[] = {desc, desc};
    VkTensorDescriptionKHR ext_outputs[] = {desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 2,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 7: Multi-node chain (conv -> batchnorm -> relu -> pool)         */
/* ------------------------------------------------------------------ */

static int test_multi_node_chain(void)
{
    uint32_t conv_in[] = {1, 64, 56, 56};
    uint32_t conv_out[] = {1, 128, 56, 56};
    uint32_t pool_out[] = {1, 128, 28, 28};

    VkTensorDescriptionKHR ext_in_desc, ext_out_desc, internal_desc;
    make_tensor_desc(&ext_in_desc, conv_in, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&internal_desc, conv_out, 4, VK_FORMAT_R16_SFLOAT, 0);
    make_tensor_desc(&ext_out_desc, pool_out, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescConvolutionKHR conv_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
        .pNext = NULL,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .kernelWidth = 3,
        .kernelHeight = 3,
        .strideX = 1,
        .strideY = 1,
        .dilationX = 1,
        .dilationY = 1,
        .paddingMode = VK_ML_PADDING_MODE_SAME_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
        .groupCount = 1,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLPrimitiveDescNormalizationKHR norm_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR,
        .pNext = NULL,
        .normType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR,
        .epsilon = 1e-5f,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        .activationType = VK_ML_ACTIVATION_FUNCTION_RELU_KHR,
        .param0 = 0.0f,
        .param1 = 0.0f,
    };

    VkMLPrimitiveDescPoolingKHR pool_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 2,
        .windowHeight = 2,
        .strideX = 2,
        .strideY = 2,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
    };

    VkMLTensorBindingKHR conv_inputs[1], conv_outputs[1];
    make_tensor_binding_external_input(&conv_inputs[0], 0, &ext_in_desc);
    make_tensor_binding_internal(&conv_outputs[0], 0, 0, &internal_desc);

    VkMLTensorBindingKHR norm_inputs[1], norm_outputs[1];
    make_tensor_binding_internal(&norm_inputs[0], 0, 0, &internal_desc);
    make_tensor_binding_internal(&norm_outputs[0], 1, 0, &internal_desc);

    VkMLTensorBindingKHR act_inputs[1], act_outputs[1];
    make_tensor_binding_internal(&act_inputs[0], 1, 0, &internal_desc);
    make_tensor_binding_internal(&act_outputs[0], 2, 0, &internal_desc);

    VkMLTensorBindingKHR pool_inputs[1], pool_outputs[1];
    make_tensor_binding_internal(&pool_inputs[0], 2, 0, &internal_desc);
    make_tensor_binding_external_output(&pool_outputs[0], 0, &ext_out_desc);

    VkMLGraphNodeCreateInfoKHR nodes[4] = {
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
            .pOperationDesc = &conv_desc,
            .inputCount = 1,
            .pInputs = conv_inputs,
            .outputCount = 1,
            .pOutputs = conv_outputs,
            .pNodeName = "conv",
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR,
            .pOperationDesc = &norm_desc,
            .inputCount = 1,
            .pInputs = norm_inputs,
            .outputCount = 1,
            .pOutputs = norm_outputs,
            .pNodeName = "batchnorm",
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = &act_desc,
            .inputCount = 1,
            .pInputs = act_inputs,
            .outputCount = 1,
            .pOutputs = act_outputs,
            .pNodeName = "relu",
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
            .pOperationDesc = &pool_desc,
            .inputCount = 1,
            .pInputs = pool_inputs,
            .outputCount = 1,
            .pOutputs = pool_outputs,
            .pNodeName = "pool",
        },
    };

    VkTensorDescriptionKHR ext_inputs[] = {ext_in_desc};
    VkTensorDescriptionKHR ext_outputs[] = {ext_out_desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 4,
        .pNodes = nodes,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 8: Scratch memory requirement query                             */
/* ------------------------------------------------------------------ */

static int test_scratch_memory_requirements(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 128, 56, 56};
    VkTensorDescriptionKHR in_desc, out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescConvolutionKHR conv_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
        .pNext = NULL,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .kernelWidth = 3,
        .kernelHeight = 3,
        .strideX = 1,
        .strideY = 1,
        .dilationX = 1,
        .dilationY = 1,
        .paddingMode = VK_ML_PADDING_MODE_SAME_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
        .groupCount = 1,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[1], outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
        .pOperationDesc = &conv_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = NULL,
    };

    VkTensorDescriptionKHR ext_inputs[] = {in_desc};
    VkTensorDescriptionKHR ext_outputs[] = {out_desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }

    VkMemoryRequirements2 mem_req = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetMLGraphMemoryRequirementsKHR(VK_NULL_HANDLE, graph, &mem_req);

    int ok = (mem_req.memoryRequirements.size > 0 &&
              mem_req.memoryRequirements.alignment == VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN);

    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Test 9: Graph destroy (including VK_NULL_HANDLE)                   */
/* ------------------------------------------------------------------ */

static int test_graph_destroy(void)
{
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, NULL);
    /* Must not crash; VK_NULL_HANDLE is a no-op */

    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        .activationType = VK_ML_ACTIVATION_FUNCTION_RELU_KHR,
        .param0 = 0.0f,
        .param1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[1], outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &desc);
    make_tensor_binding_external_output(&outputs[0], 0, &desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = &act_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = NULL,
    };

    VkTensorDescriptionKHR ext_inputs[] = {desc};
    VkTensorDescriptionKHR ext_outputs[] = {desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 10: Graph with multiple external inputs/outputs                */
/* ------------------------------------------------------------------ */

static int test_multiple_external_io(void)
{
    uint32_t a_dims[] = {1, 64, 56, 56};
    uint32_t b_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR a_desc, b_desc, out_desc;
    make_tensor_desc(&a_desc, a_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&b_desc, b_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescElementwiseKHR elem_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR,
        .pNext = NULL,
        .opType = VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[2];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &a_desc);
    make_tensor_binding_external_input(&inputs[1], 1, &b_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR,
        .pOperationDesc = &elem_desc,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "add",
    };

    VkTensorDescriptionKHR ext_inputs[] = {a_desc, b_desc};
    VkTensorDescriptionKHR ext_outputs[] = {out_desc};

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 2,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &create_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        return 1;
    }

    VkMemoryRequirements2 mem_req = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetMLGraphMemoryRequirementsKHR(VK_NULL_HANDLE, graph, &mem_req);
    if (mem_req.memoryRequirements.size == 0) {
        vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
        return 1;
    }

    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_single_node_convolution);
    RUN_TEST(test_single_node_gemm);
    RUN_TEST(test_single_node_pooling);
    RUN_TEST(test_single_node_activation);
    RUN_TEST(test_single_node_normalization);
    RUN_TEST(test_single_node_elementwise);
    RUN_TEST(test_multi_node_chain);
    RUN_TEST(test_scratch_memory_requirements);
    RUN_TEST(test_graph_destroy);
    RUN_TEST(test_multiple_external_io);

    if (g_fail_count > 0) {
        printf("\n%d test(s) FAILED\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests PASSED\n");
    return 0;
}
