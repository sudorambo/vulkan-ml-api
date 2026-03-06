/**
 * @file test_ml_graph.c
 * @brief ML graph CTS tests - creation, compilation, memory requirements.
 *
 * Verifies single-node graphs per operation type, multi-node chains,
 * scratch memory queries, graph destroy, and multiple external I/O.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "internal.h"
#include "test_helpers.h"
#include <stdio.h>
#include <string.h>

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

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
/* Test 7a: Single-node deconvolution graph                             */
/* ------------------------------------------------------------------ */

static int test_single_node_deconvolution(void)
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
        .operationType = VK_ML_OPERATION_TYPE_DECONVOLUTION_2D_KHR,
        .pOperationDesc = &conv_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "deconv",
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
/* Test 7b: Single-node fully connected graph                           */
/* ------------------------------------------------------------------ */

static int test_single_node_fully_connected(void)
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
        .operationType = VK_ML_OPERATION_TYPE_FULLY_CONNECTED_KHR,
        .pOperationDesc = &gemm_desc,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "fc",
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
/* Test 7c: Single-node average pool graph                              */
/* ------------------------------------------------------------------ */

static int test_single_node_average_pool(void)
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
        .poolType = VK_ML_OPERATION_TYPE_AVERAGE_POOL_2D_KHR,
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
        .operationType = VK_ML_OPERATION_TYPE_AVERAGE_POOL_2D_KHR,
        .pOperationDesc = &pool_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "avgpool",
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
/* Test 7d: Single-node global average pool graph                       */
/* ------------------------------------------------------------------ */

static int test_single_node_global_avg_pool(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 64, 1, 1};
    VkTensorDescriptionKHR in_desc, out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescPoolingKHR pool_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 0,
        .windowHeight = 0,
        .strideX = 0,
        .strideY = 0,
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
        .operationType = VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR,
        .pOperationDesc = &pool_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "globalavgpool",
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
/* Test 7e: Single-node sigmoid graph                                   */
/* ------------------------------------------------------------------ */

static int test_single_node_sigmoid(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        .activationType = VK_ML_ACTIVATION_FUNCTION_SIGMOID_KHR,
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
        .operationType = VK_ML_OPERATION_TYPE_SIGMOID_KHR,
        .pOperationDesc = &act_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "sigmoid",
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
/* Test 7f: Single-node tanh graph                                      */
/* ------------------------------------------------------------------ */

static int test_single_node_tanh(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        .activationType = VK_ML_ACTIVATION_FUNCTION_TANH_KHR,
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
        .operationType = VK_ML_OPERATION_TYPE_TANH_KHR,
        .pOperationDesc = &act_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "tanh",
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
/* Test 7g: Single-node leaky ReLU graph                                */
/* ------------------------------------------------------------------ */

static int test_single_node_leaky_relu(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        .activationType = VK_ML_ACTIVATION_FUNCTION_LEAKY_RELU_KHR,
        .param0 = 0.01f,
        .param1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &desc);
    make_tensor_binding_external_output(&outputs[0], 0, &desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_LEAKY_RELU_KHR,
        .pOperationDesc = &act_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "leaky_relu",
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
/* Test 7h: Single-node PReLU graph                                     */
/* ------------------------------------------------------------------ */

static int test_single_node_prelu(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        /* PReLU reuses LEAKY_RELU (same f(x)=x>0?x:a*x form) until a
           dedicated VK_ML_ACTIVATION_FUNCTION_PRELU_KHR enum is added. */
        .activationType = VK_ML_ACTIVATION_FUNCTION_LEAKY_RELU_KHR,
        .param0 = 0.1f,
        .param1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &desc);
    make_tensor_binding_external_output(&outputs[0], 0, &desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_PRELU_KHR,
        .pOperationDesc = &act_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "prelu",
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
/* Test 7i: Single-node softmax graph                                    */
/* ------------------------------------------------------------------ */

static int test_single_node_softmax(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescActivationKHR act_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR,
        .pNext = NULL,
        .activationType = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
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
        .operationType = VK_ML_OPERATION_TYPE_SOFTMAX_KHR,
        .pOperationDesc = &act_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "softmax",
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
/* Test 7j: Single-node LRN graph                                        */
/* ------------------------------------------------------------------ */

static int test_single_node_lrn(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescNormalizationKHR norm_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR,
        .pNext = NULL,
        .normType = VK_ML_OPERATION_TYPE_LRN_KHR,
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
        .operationType = VK_ML_OPERATION_TYPE_LRN_KHR,
        .pOperationDesc = &norm_desc,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "lrn",
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
/* Test 7k: Single-node elementwise mul graph                            */
/* ------------------------------------------------------------------ */

static int test_single_node_elementwise_mul(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR desc;
    make_tensor_desc(&desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLPrimitiveDescElementwiseKHR elem_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR,
        .pNext = NULL,
        .opType = VK_ML_OPERATION_TYPE_ELEMENTWISE_MUL_KHR,
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
        .operationType = VK_ML_OPERATION_TYPE_ELEMENTWISE_MUL_KHR,
        .pOperationDesc = &elem_desc,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "mul",
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
/* Test 7l: Single-node concat graph (2 inputs, 1 output)                */
/* ------------------------------------------------------------------ */

static int test_single_node_concat(void)
{
    uint32_t in1_dims[] = {1, 32, 56, 56};
    uint32_t in2_dims[] = {1, 32, 56, 56};
    uint32_t out_dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR in1_desc, in2_desc, out_desc;
    make_tensor_desc(&in1_desc, in1_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&in2_desc, in2_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLTensorBindingKHR inputs[2];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in1_desc);
    make_tensor_binding_external_input(&inputs[1], 1, &in2_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_CONCAT_KHR,
        .pOperationDesc = NULL,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "concat",
    };

    VkTensorDescriptionKHR ext_inputs[] = {in1_desc, in2_desc};
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
/* Test 7m: Single-node reshape graph                                   */
/* ------------------------------------------------------------------ */

static int test_single_node_reshape(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 64, 3136};
    VkTensorDescriptionKHR in_desc, out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 3, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RESHAPE_KHR,
        .pOperationDesc = NULL,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "reshape",
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
/* Test 7n: Single-node transpose graph                                 */
/* ------------------------------------------------------------------ */

static int test_single_node_transpose(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 56, 56, 64};
    VkTensorDescriptionKHR in_desc, out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_TRANSPOSE_KHR,
        .pOperationDesc = NULL,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "transpose",
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
/* Test 7o: Single-node resize graph                                    */
/* ------------------------------------------------------------------ */

static int test_single_node_resize(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 64, 112, 112};
    VkTensorDescriptionKHR in_desc, out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLTensorBindingKHR inputs[1];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RESIZE_KHR,
        .pOperationDesc = NULL,
        .inputCount = 1,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "resize",
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
/* Test 8: Multi-node chain (conv -> batchnorm -> relu -> pool)         */
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
/* Deep-copy ownership tests (C1 remediation)                         */
/* ------------------------------------------------------------------ */

static int test_graph_node_deep_copy(void)
{
    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r;

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
            .kernelWidth = 3, .kernelHeight = 3,
            .strideX = 1, .strideY = 1,
            .dilationX = 1, .dilationY = 1,
            .paddingMode = VK_ML_PADDING_MODE_SAME_KHR,
            .paddingTop = 0, .paddingBottom = 0,
            .paddingLeft = 0, .paddingRight = 0,
            .groupCount = 1,
            .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
            .activationParam0 = 0.0f, .activationParam1 = 0.0f,
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
            .inputCount = 1, .pInputs = inputs,
            .outputCount = 1, .pOutputs = outputs,
            .pNodeName = "conv_test",
        };

        VkTensorDescriptionKHR ext_in[] = {in_desc};
        VkTensorDescriptionKHR ext_out[] = {out_desc};

        VkMLGraphCreateInfoKHR ci = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
            .pNext = NULL, .flags = 0,
            .nodeCount = 1, .pNodes = &node,
            .externalInputCount = 1, .pExternalInputDescriptions = ext_in,
            .externalOutputCount = 1, .pExternalOutputDescriptions = ext_out,
            .constantWeightCount = 0, .pConstantWeightDescriptions = NULL,
        };

        r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &ci, NULL, &graph);
        if (r != VK_SUCCESS)
            return 1;
    }
    /* All stack locals (conv_desc, bindings, dims, node) are now out of scope.
       If the graph didn't deep-copy, these would be dangling pointers. */

    VkMemoryRequirements2 mem_req = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetMLGraphMemoryRequirementsKHR(VK_NULL_HANDLE, graph, &mem_req);

    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

static int test_graph_node_null_desc_ops(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR in_desc;
    VkTensorDescriptionKHR out_desc;
    make_tensor_desc(&in_desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLOperationTypeKHR null_desc_ops[] = {
        VK_ML_OPERATION_TYPE_CONCAT_KHR,
        VK_ML_OPERATION_TYPE_RESHAPE_KHR,
        VK_ML_OPERATION_TYPE_TRANSPOSE_KHR,
        VK_ML_OPERATION_TYPE_RESIZE_KHR,
    };

    for (int k = 0; k < 4; k++) {
        VkMLTensorBindingKHR input;
        VkMLTensorBindingKHR output;
        make_tensor_binding_external_input(&input, 0, &in_desc);
        make_tensor_binding_external_output(&output, 0, &out_desc);

        VkMLGraphNodeCreateInfoKHR node = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = null_desc_ops[k],
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &input,
            .outputCount = 1, .pOutputs = &output,
            .pNodeName = NULL,
        };

        VkMLGraphCreateInfoKHR ci = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
            .pNext = NULL, .flags = 0,
            .nodeCount = 1, .pNodes = &node,
            .externalInputCount = 1, .pExternalInputDescriptions = &in_desc,
            .externalOutputCount = 1, .pExternalOutputDescriptions = &out_desc,
            .constantWeightCount = 0, .pConstantWeightDescriptions = NULL,
        };

        VkMLGraphKHR graph = VK_NULL_HANDLE;
        VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &ci, NULL, &graph);
        if (r != VK_SUCCESS)
            return 1;
        vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    }
    return 0;
}

static int test_graph_node_name_deep_copy(void)
{
    uint32_t dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR in_desc;
    VkTensorDescriptionKHR out_desc;
    make_tensor_desc(&in_desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    {
        char name_buf[32];
        memcpy(name_buf, "relu_layer_0", 13);

        VkMLTensorBindingKHR input;
        VkMLTensorBindingKHR output;
        make_tensor_binding_external_input(&input, 0, &in_desc);
        make_tensor_binding_external_output(&output, 0, &out_desc);

        VkMLGraphNodeCreateInfoKHR node = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &input,
            .outputCount = 1, .pOutputs = &output,
            .pNodeName = name_buf,
        };

        VkMLGraphCreateInfoKHR ci = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
            .pNext = NULL, .flags = 0,
            .nodeCount = 1, .pNodes = &node,
            .externalInputCount = 1, .pExternalInputDescriptions = &in_desc,
            .externalOutputCount = 1, .pExternalOutputDescriptions = &out_desc,
            .constantWeightCount = 0, .pConstantWeightDescriptions = NULL,
        };

        VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &ci, NULL, &graph);
        if (r != VK_SUCCESS)
            return 1;

        memset(name_buf, 0xFF, sizeof(name_buf));
    }

    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* NULL pointer argument tests                                         */
/* ------------------------------------------------------------------ */

static int test_create_graph_null_args(void)
{
    VkMLGraphCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 0,
        .pNodes = NULL,
        .externalInputCount = 0,
        .pExternalInputDescriptions = NULL,
        .externalOutputCount = 0,
        .pExternalOutputDescriptions = NULL,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r1 = vkCreateMLGraphKHR(VK_NULL_HANDLE, NULL, NULL, &graph);
    if (r1 == VK_SUCCESS)
        return 1;

    VkResult r2 = vkCreateMLGraphKHR(VK_NULL_HANDLE, &ci, NULL, NULL);
    if (r2 == VK_SUCCESS)
        return 1;

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
    RUN_TEST(test_single_node_deconvolution);
    RUN_TEST(test_single_node_fully_connected);
    RUN_TEST(test_single_node_average_pool);
    RUN_TEST(test_single_node_global_avg_pool);
    RUN_TEST(test_single_node_sigmoid);
    RUN_TEST(test_single_node_tanh);
    RUN_TEST(test_single_node_leaky_relu);
    RUN_TEST(test_single_node_prelu);
    RUN_TEST(test_single_node_softmax);
    RUN_TEST(test_single_node_lrn);
    RUN_TEST(test_single_node_elementwise_mul);
    RUN_TEST(test_single_node_concat);
    RUN_TEST(test_single_node_reshape);
    RUN_TEST(test_single_node_transpose);
    RUN_TEST(test_single_node_resize);
    RUN_TEST(test_multi_node_chain);
    RUN_TEST(test_scratch_memory_requirements);
    RUN_TEST(test_graph_destroy);
    RUN_TEST(test_multiple_external_io);
    RUN_TEST(test_graph_node_deep_copy);
    RUN_TEST(test_graph_node_null_desc_ops);
    RUN_TEST(test_graph_node_name_deep_copy);
    RUN_TEST(test_create_graph_null_args);

    if (g_fail_count > 0) {
        printf("\n%d test(s) FAILED\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests PASSED\n");
    return 0;
}
