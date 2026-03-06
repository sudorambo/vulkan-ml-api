/**
 * @file test_tensor_view.c
 * @brief CTS tests for VK_KHR_ml_primitives tensor views.
 *
 * Verifies view creation, subregions, and destruction.
 * Reference implementation without real GPU; tests call API directly.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "../../src/internal.h"
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

static int test_create_view_full_tensor(void)
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

    /* Full-tensor view: no offsets/sizes means entire tensor */
    VkTensorViewCreateInfoKHR viewInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .tensor = tensor,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 0,
        .pDimensionOffsets = NULL,
        .pDimensionSizes = NULL,
    };
    VkTensorViewKHR view = VK_NULL_HANDLE;
    r = vkCreateTensorViewKHR(VK_NULL_HANDLE, &viewInfo, NULL, &view);
    vkDestroyTensorViewKHR(VK_NULL_HANDLE, view, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return (r == VK_SUCCESS && view != VK_NULL_HANDLE) ? 0 : 1;
}

static int test_create_view_subregion(void)
{
    uint32_t dims[] = {4, 4, 4, 4};
    uint32_t offsets[] = {1, 1, 1, 1};
    uint32_t sizes[] = {2, 2, 2, 2};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
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

    /* Subregion: offset + size <= dimension for each axis */
    VkTensorViewCreateInfoKHR viewInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .tensor = tensor,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensionOffsets = offsets,
        .pDimensionSizes = sizes,
    };
    VkTensorViewKHR view = VK_NULL_HANDLE;
    r = vkCreateTensorViewKHR(VK_NULL_HANDLE, &viewInfo, NULL, &view);
    vkDestroyTensorViewKHR(VK_NULL_HANDLE, view, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return (r == VK_SUCCESS && view != VK_NULL_HANDLE) ? 0 : 1;
}

static int test_destroy_view_null(void)
{
    uint32_t dims[] = {4, 8};
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
    VkTensorCreateInfoKHR tci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkTensorKHR tensor = VK_NULL_HANDLE;
    if (vkCreateTensorKHR(VK_NULL_HANDLE, &tci, NULL, &tensor) != VK_SUCCESS)
        return 1;

    uint32_t offsets[] = {0, 0};
    uint32_t sizes[] = {2, 4};
    VkTensorViewCreateInfoKHR vci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .tensor = tensor,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensionOffsets = offsets,
        .pDimensionSizes = sizes,
    };
    VkTensorViewKHR view = VK_NULL_HANDLE;
    if (vkCreateTensorViewKHR(VK_NULL_HANDLE, &vci, NULL, &view) != VK_SUCCESS) {
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
        return 1;
    }

    vkDestroyTensorViewKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, NULL);

    VkTensorViewKHR_T *v = (VkTensorViewKHR_T *)(uintptr_t)view;
    if (v->format != VK_FORMAT_R32_SFLOAT)
        return 1;

    vkDestroyTensorViewKHR(VK_NULL_HANDLE, view, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

static int test_view_format_reinterpret(void)
{
    /* Create tensor with R16_SFLOAT (2 bytes), view with R16_SINT (same size) */
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

    /* Compatible format: same element size (2 bytes) */
    VkTensorViewCreateInfoKHR viewInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .tensor = tensor,
        .format = VK_FORMAT_R16_SINT,
        .dimensionCount = 0,
        .pDimensionOffsets = NULL,
        .pDimensionSizes = NULL,
    };
    VkTensorViewKHR view = VK_NULL_HANDLE;
    r = vkCreateTensorViewKHR(VK_NULL_HANDLE, &viewInfo, NULL, &view);
    vkDestroyTensorViewKHR(VK_NULL_HANDLE, view, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return (r == VK_SUCCESS && view != VK_NULL_HANDLE) ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_create_view_full_tensor);
    RUN_TEST(test_create_view_subregion);
    RUN_TEST(test_destroy_view_null);
    RUN_TEST(test_view_format_reinterpret);

    if (g_fail_count > 0) {
        printf("\n%d test(s) failed.\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}
