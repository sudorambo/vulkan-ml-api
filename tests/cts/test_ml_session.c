/**
 * @file test_ml_session.c
 * @brief ML session CTS tests (US3) - create/destroy, scratch memory, auto-allocation.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "../../src/internal.h"
#include <stdio.h>

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

#define MOCK_SCRATCH_MEMORY ((VkDeviceMemory)(uintptr_t)0xDEAD)

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

static VkMLGraphKHR create_minimal_graph(void)
{
    uint32_t in_dims[] = {1, 64, 56, 56};
    uint32_t out_dims[] = {1, 64, 56, 56};
    VkTensorDescriptionKHR in_desc;
    VkTensorDescriptionKHR out_desc;
    make_tensor_desc(&in_desc, in_dims, 4, VK_FORMAT_R16_SFLOAT,
                     VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR);
    make_tensor_desc(&out_desc, out_dims, 4, VK_FORMAT_R16_SFLOAT,
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
    make_tensor_binding_external_input(&inputs[0], 0, &in_desc);
    make_tensor_binding_external_output(&outputs[0], 0, &out_desc);

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
    if (r != VK_SUCCESS || graph == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;
    return graph;
}

/* ------------------------------------------------------------------ */
/* test_session_create_with_scratch                                    */
/* ------------------------------------------------------------------ */

static int test_session_create_with_scratch(void)
{
    VkMLGraphKHR graph = create_minimal_graph();
    if (graph == VK_NULL_HANDLE)
        return 1;

    VkMemoryRequirements2 mem_req = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetMLGraphMemoryRequirementsKHR(VK_NULL_HANDLE, graph, &mem_req);
    VkDeviceSize scratch_size = mem_req.memoryRequirements.size;
    if (scratch_size == 0) {
        vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
        return 1;
    }

    VkMLSessionCreateInfoKHR session_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = graph,
        .scratchMemory = MOCK_SCRATCH_MEMORY,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = scratch_size,
    };

    VkMLSessionKHR session = VK_NULL_HANDLE;
    VkResult r = vkCreateMLSessionKHR(VK_NULL_HANDLE, &session_info, NULL, &session);

    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    if (session != VK_NULL_HANDLE)
        vkDestroyMLSessionKHR(VK_NULL_HANDLE, session, NULL);

    return (r == VK_SUCCESS && session != VK_NULL_HANDLE) ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* test_session_auto_allocation                                        */
/* ------------------------------------------------------------------ */

static int test_session_auto_allocation(void)
{
    VkMLGraphKHR graph = create_minimal_graph();
    if (graph == VK_NULL_HANDLE)
        return 1;

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
    VkResult r = vkCreateMLSessionKHR(VK_NULL_HANDLE, &session_info, NULL, &session);
    if (r != VK_SUCCESS || session == VK_NULL_HANDLE) {
        vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
        return 1;
    }

    VkMLSessionKHR_T *s = (VkMLSessionKHR_T *)(uintptr_t)session;
    int ok = (s->autoAllocated == VK_TRUE);

    vkDestroyMLSessionKHR(VK_NULL_HANDLE, session, NULL);
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);

    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* test_session_destroy                                                */
/* ------------------------------------------------------------------ */

static int test_session_destroy(void)
{
    VkMLGraphKHR graph = create_minimal_graph();
    if (graph == VK_NULL_HANDLE)
        return 1;

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
        .scratchMemory = MOCK_SCRATCH_MEMORY,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = mem_req.memoryRequirements.size,
    };

    VkMLSessionKHR session = VK_NULL_HANDLE;
    VkResult r = vkCreateMLSessionKHR(VK_NULL_HANDLE, &session_info, NULL, &session);
    if (r != VK_SUCCESS || session == VK_NULL_HANDLE) {
        vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
        return 1;
    }

    vkDestroyMLSessionKHR(VK_NULL_HANDLE, session, NULL);
    vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_session_destroy_null                                           */
/* ------------------------------------------------------------------ */

static int test_session_destroy_null(void)
{
    vkDestroyMLSessionKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_session_create_with_scratch);
    RUN_TEST(test_session_auto_allocation);
    RUN_TEST(test_session_destroy);
    RUN_TEST(test_session_destroy_null);

    if (g_fail_count > 0) {
        printf("\n%d test(s) FAILED\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests PASSED\n");
    return 0;
}
