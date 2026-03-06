/**
 * @file test_synchronization.c
 * @brief Synchronization CTS tests (US4) - tensor memory barriers.
 *
 * Verifies VkTensorMemoryBarrierKHR and VkTensorDependencyInfoKHR structures
 * can be correctly constructed and barrier-related constants are defined.
 * No real vkCmdPipelineBarrier2; tests validate data structures only.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern VkBool32 vk_ml_validate_tensor_memory_barrier(const VkTensorMemoryBarrierKHR *barrier);
extern VkBool32 vk_ml_validate_tensor_dependency_info(const VkTensorDependencyInfoKHR *depInfo);

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

/* ------------------------------------------------------------------ */
/* test_barrier_structure                                              */
/* ------------------------------------------------------------------ */

static int test_barrier_structure(void)
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
    VkTensorKHR tensor = VK_NULL_HANDLE;
    if (vkCreateTensorKHR(VK_NULL_HANDLE, &ci, NULL, &tensor) != VK_SUCCESS)
        return 1;

    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = tensor,
    };

    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_TRUE)
        return 1;

    barrier.tensor = VK_NULL_HANDLE;
    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_FALSE)
        return 1;

    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_dependency_info                                                */
/* ------------------------------------------------------------------ */

static int test_dependency_info(void)
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
    VkTensorKHR tensor = VK_NULL_HANDLE;
    if (vkCreateTensorKHR(VK_NULL_HANDLE, &ci, NULL, &tensor) != VK_SUCCESS)
        return 1;

    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = tensor,
    };

    VkTensorDependencyInfoKHR depInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR,
        .pNext = NULL,
        .tensorMemoryBarrierCount = 1,
        .pTensorMemoryBarriers = &barrier,
    };

    if (vk_ml_validate_tensor_dependency_info(&depInfo) != VK_TRUE)
        return 1;

    depInfo.tensorMemoryBarrierCount = 0;
    if (vk_ml_validate_tensor_dependency_info(&depInfo) != VK_FALSE)
        return 1;

    depInfo.tensorMemoryBarrierCount = 1;
    depInfo.pTensorMemoryBarriers = NULL;
    if (vk_ml_validate_tensor_dependency_info(&depInfo) != VK_FALSE)
        return 1;

    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_barrier_with_tensor                                             */
/* ------------------------------------------------------------------ */

static int test_barrier_with_tensor(void)
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
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
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

    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = tensor,
    };

    if (barrier.tensor != tensor)
        return 1;

    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_barrier_access_masks                                            */
/* ------------------------------------------------------------------ */

static int test_barrier_access_masks(void)
{
    if (VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR != 0x100000000ULL)
        return 1;
    if (VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR != 0x200000000ULL)
        return 1;

    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = VK_NULL_HANDLE,
    };

    if ((barrier.srcAccessMask & VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR) == 0)
        return 1;
    if ((barrier.dstAccessMask & VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR) == 0)
        return 1;

    return 0;
}

/* ------------------------------------------------------------------ */
/* test_barrier_validation_valid                                        */
/* ------------------------------------------------------------------ */

static int test_barrier_validation_valid(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = (VkTensorKHR)(uintptr_t)1,
    };

    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_TRUE)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_barrier_null_tensor_validation                                  */
/* ------------------------------------------------------------------ */

static int test_barrier_null_tensor_validation(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = VK_NULL_HANDLE,
    };

    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_FALSE)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_barrier_asymmetric_queue_family                                */
/* ------------------------------------------------------------------ */

static int test_barrier_asymmetric_queue_family(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = 0,
        .tensor = (VkTensorKHR)(uintptr_t)1,
    };

    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_FALSE)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_dependency_info_validation                                      */
/* ------------------------------------------------------------------ */

static int test_dependency_info_validation(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = (VkTensorKHR)(uintptr_t)1,
    };

    VkTensorDependencyInfoKHR depInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR,
        .pNext = NULL,
        .tensorMemoryBarrierCount = 1,
        .pTensorMemoryBarriers = &barrier,
    };

    if (vk_ml_validate_tensor_dependency_info(&depInfo) != VK_TRUE)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_queue_family_transfer                                           */
/* ------------------------------------------------------------------ */

static int test_queue_family_transfer(void)
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
    VkTensorKHR tensor = VK_NULL_HANDLE;
    if (vkCreateTensorKHR(VK_NULL_HANDLE, &ci, NULL, &tensor) != VK_SUCCESS)
        return 1;

    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR,
        .srcQueueFamilyIndex = 0,
        .dstQueueFamilyIndex = 1,
        .tensor = tensor,
    };

    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_TRUE)
        return 1;

    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_FALSE)
        return 1;

    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    if (vk_ml_validate_tensor_memory_barrier(&barrier) != VK_TRUE)
        return 1;

    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_barrier_structure);
    RUN_TEST(test_dependency_info);
    RUN_TEST(test_barrier_with_tensor);
    RUN_TEST(test_barrier_access_masks);
    RUN_TEST(test_queue_family_transfer);
    RUN_TEST(test_barrier_validation_valid);
    RUN_TEST(test_barrier_null_tensor_validation);
    RUN_TEST(test_barrier_asymmetric_queue_family);
    RUN_TEST(test_dependency_info_validation);

    if (g_fail_count > 0) {
        printf("\n%d test(s) failed.\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}
