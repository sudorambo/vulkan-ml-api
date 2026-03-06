/**
 * @file test_oom.c
 * @brief OOM (out-of-host-memory) tests for all vkCreate* functions.
 *
 * Uses a custom VkAllocationCallbacks that returns NULL after a
 * configurable number of successful allocations. Verifies that
 * each create function returns VK_ERROR_OUT_OF_HOST_MEMORY and
 * frees all partially-allocated resources (no leaks).
 */

#include <vulkan/vulkan_ml_primitives.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Failing allocator infrastructure                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t alloc_count;
    uint32_t fail_after;
    uint32_t free_count;
} FailingAllocatorData;

static void *VKAPI_PTR failing_alloc(void *pUserData, size_t size,
                                      size_t alignment,
                                      VkSystemAllocationScope scope)
{
    (void)alignment;
    (void)scope;
    FailingAllocatorData *data = (FailingAllocatorData *)pUserData;
    data->alloc_count++;
    if (data->alloc_count > data->fail_after)
        return NULL;
    return malloc(size);
}

static void *VKAPI_PTR failing_realloc(void *pUserData, void *pOriginal,
                                        size_t size, size_t alignment,
                                        VkSystemAllocationScope scope)
{
    (void)pUserData;
    (void)pOriginal;
    (void)size;
    (void)alignment;
    (void)scope;
    return NULL;
}

static void VKAPI_PTR failing_free(void *pUserData, void *pMemory)
{
    if (!pMemory)
        return;
    FailingAllocatorData *data = (FailingAllocatorData *)pUserData;
    data->free_count++;
    free(pMemory);
}

static VkAllocationCallbacks make_failing_allocator(FailingAllocatorData *data)
{
    VkAllocationCallbacks cb;
    memset(&cb, 0, sizeof(cb));
    cb.pUserData = data;
    cb.pfnAllocation = failing_alloc;
    cb.pfnReallocation = failing_realloc;
    cb.pfnFree = failing_free;
    return cb;
}

static void reset_failing_allocator(FailingAllocatorData *data,
                                    uint32_t fail_after)
{
    data->alloc_count = 0;
    data->fail_after = fail_after;
    data->free_count = 0;
}

/* ------------------------------------------------------------------ */
/* Test helpers                                                        */
/* ------------------------------------------------------------------ */

static int g_pass_count;
static int g_fail_count;

#define TEST_ASSERT(cond, msg)                                          \
    do {                                                                \
        if (!(cond)) {                                                  \
            printf("  FAIL: %s (line %d)\n", (msg), __LINE__);         \
            g_fail_count++;                                             \
            return;                                                     \
        }                                                               \
    } while (0)

#define SENTINEL_HANDLE ((uintptr_t)0xDEADBEEF)

/* ------------------------------------------------------------------ */
/* test_tensor_create_oom                                              */
/* ------------------------------------------------------------------ */

static void test_tensor_create_oom(void)
{
    printf("test_tensor_create_oom\n");

    uint32_t dims[] = {4, 8};
    VkDeviceSize strides[] = {32, 4};
    uint32_t qfi[] = {0};

    VkTensorDescriptionKHR desc;
    memset(&desc, 0, sizeof(desc));
    desc.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.dimensionCount = 2;
    desc.pDimensions = dims;
    desc.pStrides = strides;
    desc.usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR;

    VkTensorCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR;
    ci.pDescription = &desc;
    ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    ci.queueFamilyIndexCount = 1;
    ci.pQueueFamilyIndices = qfi;

    FailingAllocatorData fad;
    VkAllocationCallbacks cb = make_failing_allocator(&fad);

    for (uint32_t fail = 0; fail < 4; fail++) {
        reset_failing_allocator(&fad, fail);
        VkTensorKHR tensor = (VkTensorKHR)SENTINEL_HANDLE;
        VkResult result = vkCreateTensorKHR(VK_NULL_HANDLE, &ci, &cb, &tensor);

        TEST_ASSERT(result == VK_ERROR_OUT_OF_HOST_MEMORY,
                    "expected VK_ERROR_OUT_OF_HOST_MEMORY");
        TEST_ASSERT(fad.free_count == fad.alloc_count - 1,
                    "leak: free_count != alloc_count - 1");
        TEST_ASSERT((uintptr_t)tensor == SENTINEL_HANDLE,
                    "output handle was modified on failure");
    }

    reset_failing_allocator(&fad, 100);
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult result = vkCreateTensorKHR(VK_NULL_HANDLE, &ci, &cb, &tensor);
    TEST_ASSERT(result == VK_SUCCESS, "create should succeed with high limit");
    TEST_ASSERT(tensor != VK_NULL_HANDLE, "output handle should be set");
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, &cb);

    g_pass_count++;
    printf("  PASS\n");
}

/* ------------------------------------------------------------------ */
/* test_tensor_view_create_oom                                        */
/* ------------------------------------------------------------------ */

static void test_tensor_view_create_oom(void)
{
    printf("test_tensor_view_create_oom\n");

    uint32_t offsets[] = {0, 0};
    uint32_t sizes[] = {2, 4};

    VkTensorViewCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR;
    ci.tensor = (VkTensorKHR)(uintptr_t)0x1;
    ci.format = VK_FORMAT_R32_SFLOAT;
    ci.dimensionCount = 2;
    ci.pDimensionOffsets = offsets;
    ci.pDimensionSizes = sizes;

    FailingAllocatorData fad;
    VkAllocationCallbacks cb = make_failing_allocator(&fad);

    for (uint32_t fail = 0; fail < 3; fail++) {
        reset_failing_allocator(&fad, fail);
        VkTensorViewKHR view = (VkTensorViewKHR)SENTINEL_HANDLE;
        VkResult result = vkCreateTensorViewKHR(VK_NULL_HANDLE, &ci, &cb,
                                                 &view);

        TEST_ASSERT(result == VK_ERROR_OUT_OF_HOST_MEMORY,
                    "expected VK_ERROR_OUT_OF_HOST_MEMORY");
        TEST_ASSERT(fad.free_count == fad.alloc_count - 1,
                    "leak: free_count != alloc_count - 1");
        TEST_ASSERT((uintptr_t)view == SENTINEL_HANDLE,
                    "output handle was modified on failure");
    }

    reset_failing_allocator(&fad, 100);
    VkTensorViewKHR view = VK_NULL_HANDLE;
    VkResult result = vkCreateTensorViewKHR(VK_NULL_HANDLE, &ci, &cb, &view);
    TEST_ASSERT(result == VK_SUCCESS, "create should succeed with high limit");
    TEST_ASSERT(view != VK_NULL_HANDLE, "output handle should be set");
    vkDestroyTensorViewKHR(VK_NULL_HANDLE, view, &cb);

    g_pass_count++;
    printf("  PASS\n");
}

/* ------------------------------------------------------------------ */
/* test_graph_create_oom                                              */
/* ------------------------------------------------------------------ */

static void test_graph_create_oom(void)
{
    printf("test_graph_create_oom\n");

    uint32_t dims[] = {4, 8};
    VkDeviceSize strides[] = {32, 4};

    VkTensorDescriptionKHR td;
    memset(&td, 0, sizeof(td));
    td.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    td.format = VK_FORMAT_R32_SFLOAT;
    td.dimensionCount = 2;
    td.pDimensions = dims;
    td.pStrides = strides;
    td.usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR;

    VkMLTensorBindingKHR inputBinding;
    memset(&inputBinding, 0, sizeof(inputBinding));
    inputBinding.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    inputBinding.bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR;
    inputBinding.pTensorDescription = &td;

    VkTensorDescriptionKHR tdOut;
    memset(&tdOut, 0, sizeof(tdOut));
    tdOut.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    tdOut.format = VK_FORMAT_R32_SFLOAT;
    tdOut.dimensionCount = 2;
    tdOut.pDimensions = dims;
    tdOut.pStrides = strides;
    tdOut.usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR;

    VkMLTensorBindingKHR outputBinding;
    memset(&outputBinding, 0, sizeof(outputBinding));
    outputBinding.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    outputBinding.bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR;
    outputBinding.pTensorDescription = &tdOut;

    VkMLPrimitiveDescActivationKHR actDesc;
    memset(&actDesc, 0, sizeof(actDesc));
    actDesc.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR;
    actDesc.activationType = VK_ML_ACTIVATION_FUNCTION_RELU_KHR;

    VkMLGraphNodeCreateInfoKHR node;
    memset(&node, 0, sizeof(node));
    node.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR;
    node.operationType = VK_ML_OPERATION_TYPE_RELU_KHR;
    node.pOperationDesc = &actDesc;
    node.inputCount = 1;
    node.pInputs = &inputBinding;
    node.outputCount = 1;
    node.pOutputs = &outputBinding;
    node.pNodeName = "relu_node";

    VkMLGraphCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR;
    ci.nodeCount = 1;
    ci.pNodes = &node;
    ci.externalInputCount = 1;
    ci.pExternalInputDescriptions = &td;
    ci.externalOutputCount = 1;
    ci.pExternalOutputDescriptions = &tdOut;
    ci.constantWeightCount = 0;

    FailingAllocatorData fad;
    VkAllocationCallbacks cb = make_failing_allocator(&fad);

    uint32_t oom_iterations = 0;
    for (uint32_t fail = 0; fail < 100; fail++) {
        reset_failing_allocator(&fad, fail);
        VkMLGraphKHR graph = (VkMLGraphKHR)SENTINEL_HANDLE;
        VkResult result = vkCreateMLGraphKHR(VK_NULL_HANDLE, &ci, &cb, &graph);

        if (result == VK_SUCCESS) {
            vkDestroyMLGraphKHR(VK_NULL_HANDLE, graph, &cb);
            break;
        }

        oom_iterations++;
        TEST_ASSERT(result == VK_ERROR_OUT_OF_HOST_MEMORY,
                    "expected VK_ERROR_OUT_OF_HOST_MEMORY");
        TEST_ASSERT(fad.free_count == fad.alloc_count - 1,
                    "leak: free_count != alloc_count - 1");
        TEST_ASSERT((uintptr_t)graph == SENTINEL_HANDLE,
                    "output handle was modified on failure");
    }

    TEST_ASSERT(oom_iterations >= 5,
                "expected at least 5 OOM iterations for graph create");

    g_pass_count++;
    printf("  PASS (exercised %u OOM points)\n", oom_iterations);
}

/* ------------------------------------------------------------------ */
/* test_session_create_oom                                            */
/* ------------------------------------------------------------------ */

static void test_session_create_oom(void)
{
    printf("test_session_create_oom\n");

    VkMLSessionCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR;
    ci.graph = (VkMLGraphKHR)(uintptr_t)0x1;
    ci.scratchMemory = VK_NULL_HANDLE;

    FailingAllocatorData fad;
    VkAllocationCallbacks cb = make_failing_allocator(&fad);

    reset_failing_allocator(&fad, 0);
    VkMLSessionKHR session = (VkMLSessionKHR)SENTINEL_HANDLE;
    VkResult result = vkCreateMLSessionKHR(VK_NULL_HANDLE, &ci, &cb, &session);

    TEST_ASSERT(result == VK_ERROR_OUT_OF_HOST_MEMORY,
                "expected VK_ERROR_OUT_OF_HOST_MEMORY");
    TEST_ASSERT(fad.free_count == 0,
                "no successful allocs to free");
    TEST_ASSERT((uintptr_t)session == SENTINEL_HANDLE,
                "output handle was modified on failure");

    reset_failing_allocator(&fad, 100);
    session = VK_NULL_HANDLE;
    result = vkCreateMLSessionKHR(VK_NULL_HANDLE, &ci, &cb, &session);
    TEST_ASSERT(result == VK_SUCCESS, "create should succeed with high limit");
    TEST_ASSERT(session != VK_NULL_HANDLE, "output handle should be set");
    vkDestroyMLSessionKHR(VK_NULL_HANDLE, session, &cb);

    g_pass_count++;
    printf("  PASS\n");
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    g_pass_count = 0;
    g_fail_count = 0;

    test_tensor_create_oom();
    test_tensor_view_create_oom();
    test_graph_create_oom();
    test_session_create_oom();

    printf("\nOOM tests: %d passed, %d failed\n", g_pass_count, g_fail_count);
    return g_fail_count == 0 ? 0 : 1;
}
