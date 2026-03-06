/**
 * @file test_tensor_lifecycle.c
 * @brief CTS tests for VK_KHR_ml_primitives tensor lifecycle.
 *
 * Verifies tensor creation, memory requirements, binding, and destruction.
 * Reference implementation without real GPU; tests call API directly.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <assert.h>

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static int test_create_tensor_4d_fp16(void)
{
    uint32_t dims[] = {1, 3, 224, 224};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

static int test_create_tensor_8d(void)
{
    uint32_t dims[] = {1, 1, 1, 1, 1, 1, 1, 2};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 8,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

static int test_memory_requirements(void)
{
    uint32_t dims[] = {1, 3, 224, 224};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;

    VkTensorMemoryRequirementsInfoKHR reqInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
        .pNext = NULL,
        .tensor = tensor,
    };
    VkMemoryRequirements2 memReq = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetTensorMemoryRequirementsKHR(VK_NULL_HANDLE, &reqInfo, &memReq);

    int ok = (memReq.memoryRequirements.size > 0 &&
              memReq.memoryRequirements.alignment > 0);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return ok ? 0 : 1;
}

static int test_bind_memory(void)
{
    uint32_t dims[] = {1, 2, 2, 2};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;

    VkBindTensorMemoryInfoKHR bindInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
        .pNext = NULL,
        .tensor = tensor,
        .memory = (VkDeviceMemory)(uintptr_t)0x1,
        .memoryOffset = 0,
    };
    r = vkBindTensorMemoryKHR(VK_NULL_HANDLE, 1, &bindInfo);
    if (r != VK_SUCCESS)
        return 1;

    VkTensorMemoryRequirementsInfoKHR reqInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
        .pNext = NULL,
        .tensor = tensor,
    };
    VkMemoryRequirements2 memReq = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetTensorMemoryRequirementsKHR(VK_NULL_HANDLE, &reqInfo, &memReq);
    int ok = (memReq.memoryRequirements.size > 0);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return ok ? 0 : 1;
}

static int test_destroy_null_handle(void)
{
    uint32_t dims[] = {2, 3};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkTensorKHR live = VK_NULL_HANDLE;
    if (vkCreateTensorKHR(VK_NULL_HANDLE, &ci, NULL, &live) != VK_SUCCESS)
        return 1;

    vkDestroyTensorKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, NULL);

    VkTensorMemoryRequirementsInfoKHR reqInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
        .pNext = NULL,
        .tensor = live,
    };
    VkMemoryRequirements2 memReq = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetTensorMemoryRequirementsKHR(VK_NULL_HANDLE, &reqInfo, &memReq);
    if (memReq.memoryRequirements.size == 0)
        return 1;

    vkDestroyTensorKHR(VK_NULL_HANDLE, live, NULL);
    return 0;
}

static int test_create_multiple_formats(void)
{
    uint32_t dims[] = {1, 2, 2, 2};
    VkFormat formats[] = {
        VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R8_SINT,
        VK_FORMAT_R8_UINT,
    };
    const size_t n = sizeof(formats) / sizeof(formats[0]);

    for (size_t i = 0; i < n; i++) {
        VkTensorDescriptionKHR desc = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
            .pNext = NULL,
            .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
            .format = formats[i],
            .dimensionCount = 4,
            .pDimensions = dims,
            .pStrides = NULL,
            .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
        };
        VkTensorCreateInfoKHR createInfo = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
            .pNext = NULL,
            .flags = 0,
            .pDescription = &desc,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = NULL,
        };
        VkTensorKHR tensor = VK_NULL_HANDLE;
        VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);
        if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
            return 1;
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Ownership tests (H1 remediation)                                   */
/* ------------------------------------------------------------------ */

static int test_tensor_description_owns_dims(void)
{
    uint32_t dims[] = {2, 3, 4, 5};
    VkDeviceSize strides[] = {120, 40, 10, 2};
    VkDeviceSize expected_size = (VkDeviceSize)(2 * 3 * 4 * 5) * 2; /* fp16 = 2 bytes */

    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = strides,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;

    memset(dims, 0xFF, sizeof(dims));
    memset(strides, 0xFF, sizeof(strides));

    VkTensorMemoryRequirementsInfoKHR reqInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
        .pNext = NULL,
        .tensor = tensor,
    };
    VkMemoryRequirements2 memReq = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = NULL,
    };
    vkGetTensorMemoryRequirementsKHR(VK_NULL_HANDLE, &reqInfo, &memReq);

    int ok = (memReq.memoryRequirements.size == expected_size);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Allocation alignment tests (H2 remediation)                        */
/* ------------------------------------------------------------------ */

static size_t s_captured_alignment = 0;

static void *VKAPI_PTR capturing_alloc(void *pUserData, size_t size,
                                       size_t alignment,
                                       VkSystemAllocationScope scope)
{
    (void)pUserData;
    (void)scope;
    s_captured_alignment = alignment;
    return malloc(size);
}

static void VKAPI_PTR capturing_free(void *pUserData, void *pMemory)
{
    (void)pUserData;
    free(pMemory);
}

static int test_alloc_callback_alignment(void)
{
    VkAllocationCallbacks cbs = {
        .pUserData = NULL,
        .pfnAllocation = capturing_alloc,
        .pfnReallocation = NULL,
        .pfnFree = capturing_free,
        .pfnInternalAllocation = NULL,
        .pfnInternalFree = NULL,
    };

    uint32_t dims[] = {1, 2, 3, 4};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };

    s_captured_alignment = 0;
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, &cbs, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;

    int ok = (s_captured_alignment >= _Alignof(max_align_t));
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, &cbs);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_create_tensor_4d_fp16);
    RUN_TEST(test_create_tensor_8d);
    RUN_TEST(test_memory_requirements);
    RUN_TEST(test_bind_memory);
    RUN_TEST(test_destroy_null_handle);
    RUN_TEST(test_create_multiple_formats);
    RUN_TEST(test_tensor_description_owns_dims);
    RUN_TEST(test_alloc_callback_alignment);

    if (g_fail_count > 0) {
        printf("\n%d test(s) failed.\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}
