/**
 * @file test_ml_dispatch.c
 * @brief ML dispatch CTS tests (US3) - vkCmdDispatchMLGraphKHR.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "vk_ml_validation.h"
#include "test_helpers.h"
#include <stdio.h>
#include <string.h>

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

#define MOCK_COMMAND_BUFFER ((VkCommandBuffer)(uintptr_t)0xBEEF)

static VkTensorKHR create_tensor(const VkTensorDescriptionKHR *desc)
{
    VkTensorCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };

    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &create_info, NULL, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;
    return tensor;
}

/* ------------------------------------------------------------------ */
/* test_dispatch_basic                                                 */
/* ------------------------------------------------------------------ */

static int test_dispatch_basic(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 64, 56, 56};
    uint32_t weight_dims[] = {64, 64, 3, 3};
    VkTensorDescriptionKHR in_desc;
    VkTensorDescriptionKHR out_desc;
    VkTensorDescriptionKHR weight_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);
    make_tensor_desc(&weight_desc, weight_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_WEIGHT_BIT_KHR);

    VkTensorKHR input_tensor = create_tensor(&in_desc);
    VkTensorKHR output_tensor = create_tensor(&out_desc);
    VkTensorKHR weight_tensor = create_tensor(&weight_desc);
    if (input_tensor == VK_NULL_HANDLE || output_tensor == VK_NULL_HANDLE ||
        weight_tensor == VK_NULL_HANDLE) {
        if (input_tensor != VK_NULL_HANDLE) vkDestroyTensorKHR(VK_NULL_HANDLE, input_tensor, NULL);
        if (output_tensor != VK_NULL_HANDLE) vkDestroyTensorKHR(VK_NULL_HANDLE, output_tensor, NULL);
        if (weight_tensor != VK_NULL_HANDLE) vkDestroyTensorKHR(VK_NULL_HANDLE, weight_tensor, NULL);
        return 1;
    }

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

    VkMLTensorBindingKHR inputs[2];
    VkMLTensorBindingKHR outputs[1];
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_weight(&inputs[1], 0, &weight_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
        .pOperationDesc = &conv_desc,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 1,
        .pOutputs = outputs,
        .pNodeName = "conv",
    };

    VkTensorDescriptionKHR ext_inputs[] = {in_desc};
    VkTensorDescriptionKHR ext_outputs[] = {out_desc};
    VkTensorDescriptionKHR ext_weights[] = {weight_desc};

    VkMLGraphCreateInfoKHR graph_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 1,
        .pConstantWeightDescriptions = ext_weights,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &graph_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        vkDestroyTensorKHR(VK_NULL_HANDLE, input_tensor, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, output_tensor, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, weight_tensor, NULL);
        return 1;
    }

    VkMemoryRequirements2 mem_req = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetMLGraphMemoryRequirementsKHR(VK_NULL_HANDLE, graph, &mem_req);

    VkMLSessionCreateInfoKHR session_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = graph,
        .scratchMemory = (VkDeviceMemory)(uintptr_t)0xDEAD,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = mem_req.memoryRequirements.size,
    };

    VkMLSessionKHR session = VK_NULL_HANDLE;
    r = vkCreateMLSessionKHR(VK_NULL_HANDLE, &session_info, NULL, &session);
    if (r != VK_SUCCESS || session == VK_NULL_HANDLE) {
        vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, input_tensor, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, output_tensor, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, weight_tensor, NULL);
        return 1;
    }

    VkTensorKHR input_tensors[] = {input_tensor};
    VkTensorKHR output_tensors[] = {output_tensor};
    VkTensorKHR weight_tensors[] = {weight_tensor};

    VkMLGraphDispatchInfoKHR dispatch_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = session,
        .inputTensorCount = 1,
        .pInputTensors = input_tensors,
        .outputTensorCount = 1,
        .pOutputTensors = output_tensors,
        .weightTensorCount = 1,
        .pWeightTensors = weight_tensors,
    };

    vkCmdDispatchMLGraphKHR(MOCK_COMMAND_BUFFER, &dispatch_info);

    vkDestroyMLSessionKHR(VK_NULL_HANDLE, session, NULL);
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, input_tensor, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, output_tensor, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, weight_tensor, NULL);

    return 0;
}

/* ------------------------------------------------------------------ */
/* test_dispatch_multiple_ios                                          */
/* ------------------------------------------------------------------ */

static int test_dispatch_multiple_ios(void)
{
    uint32_t a_dims[] = {1, 64, 56, 56};
    uint32_t b_dims[] = {1, 64, 56, 56};
    uint32_t out1_dims[] = {1, 64, 56, 56};
    uint32_t out2_dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR a_desc;
    VkTensorDescriptionKHR b_desc;
    VkTensorDescriptionKHR out1_desc;
    VkTensorDescriptionKHR out2_desc;
    make_tensor_desc(&a_desc, a_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&b_desc, b_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out1_desc, out1_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);
    make_tensor_desc(&out2_desc, out2_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR);

    VkTensorKHR tensor_a = create_tensor(&a_desc);
    VkTensorKHR tensor_b = create_tensor(&b_desc);
    VkTensorKHR tensor_out1 = create_tensor(&out1_desc);
    VkTensorKHR tensor_out2 = create_tensor(&out2_desc);
    if (tensor_a == VK_NULL_HANDLE || tensor_b == VK_NULL_HANDLE ||
        tensor_out1 == VK_NULL_HANDLE || tensor_out2 == VK_NULL_HANDLE) {
        if (tensor_a != VK_NULL_HANDLE) vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_a, NULL);
        if (tensor_b != VK_NULL_HANDLE) vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_b, NULL);
        if (tensor_out1 != VK_NULL_HANDLE) vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out1, NULL);
        if (tensor_out2 != VK_NULL_HANDLE) vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out2, NULL);
        return 1;
    }

    VkMLPrimitiveDescElementwiseKHR elem_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR,
        .pNext = NULL,
        .opType = VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkMLTensorBindingKHR inputs[2];
    VkMLTensorBindingKHR outputs[2];
    make_tensor_binding_external_input(&inputs[0], 0, &a_desc);
    make_tensor_binding_external_input(&inputs[1], 1, &b_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out1_desc);
    make_tensor_binding_external_output(&outputs[1], 1, &out2_desc);

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR,
        .pOperationDesc = &elem_desc,
        .inputCount = 2,
        .pInputs = inputs,
        .outputCount = 2,
        .pOutputs = outputs,
        .pNodeName = "add",
    };

    VkTensorDescriptionKHR ext_inputs[] = {a_desc, b_desc};
    VkTensorDescriptionKHR ext_outputs[] = {out1_desc, out2_desc};

    VkMLGraphCreateInfoKHR graph_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 2,
        .pExternalInputDescriptions = ext_inputs,
        .externalOutputCount = 2,
        .pExternalOutputDescriptions = ext_outputs,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &graph_info, NULL, &graph);
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE) {
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_a, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_b, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out1, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out2, NULL);
        return 1;
    }

    VkMLSessionCreateInfoKHR session_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = graph,
        .scratchMemory = VK_NULL_HANDLE,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = 0,
    };

    VkMLSessionKHR session = VK_NULL_HANDLE;
    r = vkCreateMLSessionKHR(VK_NULL_HANDLE, &session_info, NULL, &session);
    if (r != VK_SUCCESS || session == VK_NULL_HANDLE) {
        vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_a, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_b, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out1, NULL);
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out2, NULL);
        return 1;
    }

    VkTensorKHR input_tensors[] = {tensor_a, tensor_b};
    VkTensorKHR output_tensors[] = {tensor_out1, tensor_out2};

    VkMLGraphDispatchInfoKHR dispatch_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = session,
        .inputTensorCount = 2,
        .pInputTensors = input_tensors,
        .outputTensorCount = 2,
        .pOutputTensors = output_tensors,
        .weightTensorCount = 0,
        .pWeightTensors = NULL,
    };

    vkCmdDispatchMLGraphKHR(MOCK_COMMAND_BUFFER, &dispatch_info);

    vkDestroyMLSessionKHR(VK_NULL_HANDLE, session, NULL);
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_a, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_b, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out1, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor_out2, NULL);

    return 0;
}

/* ------------------------------------------------------------------ */
/* NULL array pointer tests (H5 remediation)                          */
/* ------------------------------------------------------------------ */

static int test_dispatch_null_input_tensors(void)
{
    VkMLGraphDispatchInfoKHR info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = NULL,
        .outputTensorCount = 1,
        .pOutputTensors = (const VkTensorKHR[]){(VkTensorKHR)(uintptr_t)0x2},
        .weightTensorCount = 0,
        .pWeightTensors = NULL,
    };
    return vk_ml_validate_dispatch(&info) == VK_FALSE ? 0 : 1;
}

static int test_dispatch_null_output_tensors(void)
{
    VkMLGraphDispatchInfoKHR info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = (const VkTensorKHR[]){(VkTensorKHR)(uintptr_t)0x2},
        .outputTensorCount = 1,
        .pOutputTensors = NULL,
        .weightTensorCount = 0,
        .pWeightTensors = NULL,
    };
    return vk_ml_validate_dispatch(&info) == VK_FALSE ? 0 : 1;
}

static int test_dispatch_null_weight_tensors(void)
{
    VkMLGraphDispatchInfoKHR info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = (const VkTensorKHR[]){(VkTensorKHR)(uintptr_t)0x2},
        .outputTensorCount = 1,
        .pOutputTensors = (const VkTensorKHR[]){(VkTensorKHR)(uintptr_t)0x3},
        .weightTensorCount = 1,
        .pWeightTensors = NULL,
    };
    return vk_ml_validate_dispatch(&info) == VK_FALSE ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_dispatch_basic);
    RUN_TEST(test_dispatch_multiple_ios);
    RUN_TEST(test_dispatch_null_input_tensors);
    RUN_TEST(test_dispatch_null_output_tensors);
    RUN_TEST(test_dispatch_null_weight_tensors);

    if (g_fail_count > 0) {
        printf("\n%d test(s) FAILED\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests PASSED\n");
    return 0;
}
