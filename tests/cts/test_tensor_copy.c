/**
 * @file test_tensor_copy.c
 * @brief CTS tests for VK_KHR_ml_primitives tensor copy.
 *
 * Verifies vkCmdCopyTensorKHR behavior with mock command buffer.
 * Reference implementation without real GPU; tests call API directly.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include <stdio.h>
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

static int test_copy_basic(void)
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
        .usage = VK_TENSOR_USAGE_TRANSFER_SRC_BIT_KHR | VK_TENSOR_USAGE_TRANSFER_DST_BIT_KHR,
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

    VkTensorKHR src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &src);
    if (r != VK_SUCCESS || src == VK_NULL_HANDLE)
        return 1;
    desc.usage = VK_TENSOR_USAGE_TRANSFER_DST_BIT_KHR;
    r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &dst);
    if (r != VK_SUCCESS || dst == VK_NULL_HANDLE) {
        vkDestroyTensorKHR(VK_NULL_HANDLE, src, NULL);
        return 1;
    }

    uint32_t srcOff[] = {0, 0, 0, 0};
    uint32_t dstOff[] = {0, 0, 0, 0};
    uint32_t ext[] = {1, 2, 2, 2};
    VkTensorCopyKHR region = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_COPY_KHR,
        .pNext = NULL,
        .dimensionCount = 4,
        .pSrcOffsets = srcOff,
        .pDstOffsets = dstOff,
        .pExtents = ext,
    };
    VkCopyTensorInfoKHR copyInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_COPY_INFO_KHR,
        .pNext = NULL,
        .srcTensor = src,
        .dstTensor = dst,
        .regionCount = 1,
        .pRegions = &region,
    };

    VkCommandBuffer cmd = (VkCommandBuffer)(uintptr_t)0xBEEF;
    vkCmdCopyTensorKHR(cmd, &copyInfo);
    /* No crash = pass */

    vkDestroyTensorKHR(VK_NULL_HANDLE, dst, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, src, NULL);
    return 0;
}

static int test_copy_null_cmd(void)
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
        .usage = VK_TENSOR_USAGE_TRANSFER_SRC_BIT_KHR | VK_TENSOR_USAGE_TRANSFER_DST_BIT_KHR,
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

    VkTensorKHR src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &src);
    if (r != VK_SUCCESS || src == VK_NULL_HANDLE)
        return 1;
    r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &dst);
    if (r != VK_SUCCESS || dst == VK_NULL_HANDLE) {
        vkDestroyTensorKHR(VK_NULL_HANDLE, src, NULL);
        return 1;
    }

    uint32_t srcOff[] = {0, 0, 0, 0};
    uint32_t dstOff[] = {0, 0, 0, 0};
    uint32_t ext[] = {1, 2, 2, 2};
    VkTensorCopyKHR region = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_COPY_KHR,
        .pNext = NULL,
        .dimensionCount = 4,
        .pSrcOffsets = srcOff,
        .pDstOffsets = dstOff,
        .pExtents = ext,
    };
    VkCopyTensorInfoKHR copyInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_COPY_INFO_KHR,
        .pNext = NULL,
        .srcTensor = src,
        .dstTensor = dst,
        .regionCount = 1,
        .pRegions = &region,
    };

    vkCmdCopyTensorKHR(VK_NULL_HANDLE, &copyInfo);
    /* Should handle gracefully (no crash) */

    vkDestroyTensorKHR(VK_NULL_HANDLE, dst, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, src, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_copy_basic);
    RUN_TEST(test_copy_null_cmd);

    if (g_fail_count > 0) {
        printf("\n%d test(s) failed.\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}
