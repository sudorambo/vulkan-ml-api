/**
 * @file session_validation.c
 * @brief ML session creation validation for VK_KHR_ml_primitives.
 */

#include "vk_ml_validation.h"
#include "internal.h"

#include <stddef.h>

VkBool32 vk_ml_validate_session_create(
    const VkMLSessionCreateInfoKHR *pCreateInfo,
    VkDeviceSize requiredScratchSize,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!pCreateInfo || !features)
        return VK_FALSE;
    if ((int)pCreateInfo->sType != VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR)
        return VK_FALSE;

    /* VUID_SESSION_GRAPH_VALID */
    if (pCreateInfo->graph == VK_NULL_HANDLE)
        return VK_FALSE;

    if (pCreateInfo->scratchMemory != VK_NULL_HANDLE) {
        /* VUID_SESSION_SCRATCH_OFFSET_ALIGN */
        if (pCreateInfo->scratchMemoryOffset % VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN != 0)
            return VK_FALSE;
        /* VUID_SESSION_SCRATCH_SIZE */
        if (pCreateInfo->scratchMemorySize < requiredScratchSize)
            return VK_FALSE;
    } else {
        /* VUID_SESSION_SCRATCH_AUTO - NULL handle requires autoAlloc feature */
        if (!features->mlGraphScratchAutoAllocation)
            return VK_FALSE;
    }

    return VK_TRUE;
}
