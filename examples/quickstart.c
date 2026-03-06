/**
 * @file quickstart.c
 * @brief Minimal end-to-end example demonstrating the VK_KHR_ml_primitives API.
 *
 * Workflow: create tensors -> build graph -> create session -> dispatch -> cleanup.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "internal.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOCK_DEVICE_MEMORY ((VkDeviceMemory)(uintptr_t)0x1)
#define MOCK_COMMAND_BUFFER ((VkCommandBuffer)(uintptr_t)0x1)

int main(void)
{
    VkDevice device = VK_NULL_HANDLE;
    const VkAllocationCallbacks *allocator = NULL;

    /* 1-3: Create input, weight, and output tensors */
    uint32_t input_dims[] = {1, 3, 224, 224};
    uint32_t weight_dims[] = {64, 3, 3, 3};
    uint32_t output_dims[] = {1, 64, 224, 224};

    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = input_dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };

    VkTensorDescriptionKHR weight_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = weight_dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_WEIGHT_BIT_KHR,
    };

    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = output_dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    VkTensorCreateInfoKHR input_ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &input_desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };

    VkTensorCreateInfoKHR weight_ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &weight_desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };

    VkTensorCreateInfoKHR output_ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &output_desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };

    VkTensorKHR input_tensor = VK_NULL_HANDLE;
    VkTensorKHR weight_tensor = VK_NULL_HANDLE;
    VkTensorKHR output_tensor = VK_NULL_HANDLE;

    if (vkCreateTensorKHR(device, &input_ci, allocator, &input_tensor) != VK_SUCCESS ||
        vkCreateTensorKHR(device, &weight_ci, allocator, &weight_tensor) != VK_SUCCESS ||
        vkCreateTensorKHR(device, &output_ci, allocator, &output_tensor) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create tensors\n");
        return 1;
    }

    /* 6: Query memory requirements for all three tensors */
    VkMemoryRequirements2 mem_reqs = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };

    VkTensorMemoryRequirementsInfoKHR input_mem_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
        .pNext = NULL,
        .tensor = input_tensor,
    };
    vkGetTensorMemoryRequirementsKHR(device, &input_mem_info, &mem_reqs);

    VkTensorMemoryRequirementsInfoKHR weight_mem_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
        .pNext = NULL,
        .tensor = weight_tensor,
    };
    vkGetTensorMemoryRequirementsKHR(device, &weight_mem_info, &mem_reqs);

    VkTensorMemoryRequirementsInfoKHR output_mem_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
        .pNext = NULL,
        .tensor = output_tensor,
    };
    vkGetTensorMemoryRequirementsKHR(device, &output_mem_info, &mem_reqs);

    /* 7: Bind memory (mock VkDeviceMemory) */
    VkBindTensorMemoryInfoKHR bind_infos[3] = {
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
            .pNext = NULL,
            .tensor = input_tensor,
            .memory = MOCK_DEVICE_MEMORY,
            .memoryOffset = 0,
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
            .pNext = NULL,
            .tensor = weight_tensor,
            .memory = MOCK_DEVICE_MEMORY,
            .memoryOffset = 0,
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
            .pNext = NULL,
            .tensor = output_tensor,
            .memory = MOCK_DEVICE_MEMORY,
            .memoryOffset = 0,
        },
    };

    if (vkBindTensorMemoryKHR(device, 3, bind_infos) != VK_SUCCESS) {
        fprintf(stderr, "Failed to bind tensor memory\n");
        vkDestroyTensorKHR(device, input_tensor, allocator);
        vkDestroyTensorKHR(device, weight_tensor, allocator);
        vkDestroyTensorKHR(device, output_tensor, allocator);
        return 1;
    }

    /* 8: Build single-node convolution graph (3x3 kernel, stride 1, padding SAME) */
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

    VkMLTensorBindingKHR node_inputs[2] = {
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
            .pNext = NULL,
            .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR,
            .nodeIndex = 0,
            .tensorIndex = 0,
            .pTensorDescription = &input_desc,
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
            .pNext = NULL,
            .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_WEIGHT_KHR,
            .nodeIndex = 0,
            .tensorIndex = 0,
            .pTensorDescription = &weight_desc,
        },
    };

    VkMLTensorBindingKHR node_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
        .pOperationDesc = &conv_desc,
        .inputCount = 2,
        .pInputs = node_inputs,
        .outputCount = 1,
        .pOutputs = &node_output,
        .pNodeName = "conv2d",
    };

    VkTensorDescriptionKHR ext_inputs[] = {input_desc};
    VkTensorDescriptionKHR ext_outputs[] = {output_desc};
    VkTensorDescriptionKHR ext_weights[] = {weight_desc};

    VkMLGraphCreateInfoKHR graph_ci = {
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
    if (vkCreateMLGraphKHR(device, &graph_ci, allocator, &graph) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create ML graph\n");
        vkDestroyTensorKHR(device, input_tensor, allocator);
        vkDestroyTensorKHR(device, weight_tensor, allocator);
        vkDestroyTensorKHR(device, output_tensor, allocator);
        return 1;
    }

    /* 9: Create ML session with mock scratch memory */
    VkMemoryRequirements2 scratch_reqs = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetMLGraphMemoryRequirementsKHR(device, graph, &scratch_reqs);

    VkMLSessionCreateInfoKHR session_ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = graph,
        .scratchMemory = MOCK_DEVICE_MEMORY,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = scratch_reqs.memoryRequirements.size,
    };

    VkMLSessionKHR session = VK_NULL_HANDLE;
    if (vkCreateMLSessionKHR(device, &session_ci, allocator, &session) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create ML session\n");
        vkDestroyMLGraphKHR(device, graph, allocator);
        vkDestroyTensorKHR(device, input_tensor, allocator);
        vkDestroyTensorKHR(device, weight_tensor, allocator);
        vkDestroyTensorKHR(device, output_tensor, allocator);
        return 1;
    }

    /* 10: Call vkCmdDispatchMLGraphKHR with mock command buffer */
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

    /* 11: Clean up all resources in reverse order */
    vkDestroyMLSessionKHR(device, session, allocator);
    vkDestroyMLGraphKHR(device, graph, allocator);
    vkDestroyTensorKHR(device, output_tensor, allocator);
    vkDestroyTensorKHR(device, weight_tensor, allocator);
    vkDestroyTensorKHR(device, input_tensor, allocator);

    printf("Quickstart example completed successfully.\n");
    return 0;
}
