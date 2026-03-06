/**
 * @file test_tensor_lifecycle.c
 * @brief CTS tests for VK_KHR_ml_primitives tensor lifecycle.
 *
 * Verifies tensor creation, memory requirements, binding, and destruction.
 * Reference implementation without real GPU; tests call API directly.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
              memReq.memoryRequirements.alignment == VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN);
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

    VkTensorKHR_T* t = (VkTensorKHR_T*)(uintptr_t)tensor;
    int ok = (t->memoryBound == VK_TRUE);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return ok ? 0 : 1;
}

static int test_destroy_null_handle(void)
{
    vkDestroyTensorKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, NULL);
    return 0; /* No crash = pass */
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

    if (g_fail_count > 0) {
        printf("\n%d test(s) failed.\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}
