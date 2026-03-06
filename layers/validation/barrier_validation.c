/**
 * @file barrier_validation.c
 * @brief Tensor memory barrier validation for VK_KHR_ml_primitives.
 */

#include "../validation/vk_ml_validation.h"

/* Valid access mask bits for tensor memory barriers */
static const VkAccessFlags2 VALID_TENSOR_ACCESS_MASK =
    VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR |
    VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR |
    VK_ACCESS_2_SHADER_READ_BIT |
    VK_ACCESS_2_SHADER_WRITE_BIT |
    VK_ACCESS_2_TRANSFER_READ_BIT |
    VK_ACCESS_2_TRANSFER_WRITE_BIT;

VkBool32 vk_ml_validate_tensor_memory_barrier(
    const VkTensorMemoryBarrierKHR *barrier)
{
    if (!barrier)
        return VK_FALSE;

    /* Tensor handle must be valid */
    if (barrier->tensor == VK_NULL_HANDLE)
        return VK_FALSE;

    /* Access masks must only use valid bits */
    if ((barrier->srcAccessMask & ~VALID_TENSOR_ACCESS_MASK) != 0)
        return VK_FALSE;
    if ((barrier->dstAccessMask & ~VALID_TENSOR_ACCESS_MASK) != 0)
        return VK_FALSE;

    /* Queue family indices: both IGNORED or both valid */
    VkBool32 srcIgnored = (barrier->srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    VkBool32 dstIgnored = (barrier->dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    if (srcIgnored != dstIgnored)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_tensor_dependency_info(
    const VkTensorDependencyInfoKHR *depInfo)
{
    if (!depInfo)
        return VK_FALSE;

    if (depInfo->tensorMemoryBarrierCount == 0)
        return VK_FALSE;

    if (!depInfo->pTensorMemoryBarriers)
        return VK_FALSE;

    for (uint32_t i = 0; i < depInfo->tensorMemoryBarrierCount; i++) {
        if (!vk_ml_validate_tensor_memory_barrier(
                &depInfo->pTensorMemoryBarriers[i]))
            return VK_FALSE;
    }

    return VK_TRUE;
}
