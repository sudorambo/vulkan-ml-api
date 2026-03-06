/**
 * @file ml_session.c
 * @brief VK_KHR_ml_primitives ML session lifecycle implementation.
 */

#include "internal.h"

/* ------------------------------------------------------------------ */
/* ML session creation and destruction                                */
/* ------------------------------------------------------------------ */

VKAPI_ATTR VkResult VKAPI_CALL vkCreateMLSessionKHR(
    VkDevice                           device,
    const VkMLSessionCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*       pAllocator,
    VkMLSessionKHR*                    pSession)
{
    (void)device;
    if (!pCreateInfo || !pSession)
        return VK_ERROR_UNKNOWN;
    if (pCreateInfo->graph == VK_NULL_HANDLE)
        return VK_ERROR_UNKNOWN;

    VkMLSessionKHR_T *session = (VkMLSessionKHR_T *)vk_ml_alloc(pAllocator,
        sizeof(VkMLSessionKHR_T));
    if (!session)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    session->graph = pCreateInfo->graph;
    session->scratchMemory = pCreateInfo->scratchMemory;
    session->scratchMemoryOffset = pCreateInfo->scratchMemoryOffset;
    session->scratchMemorySize = pCreateInfo->scratchMemorySize;
    session->autoAllocated = (pCreateInfo->scratchMemory == VK_NULL_HANDLE) ?
        VK_TRUE : VK_FALSE;

    *pSession = (VkMLSessionKHR)(uintptr_t)session;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyMLSessionKHR(
    VkDevice                        device,
    VkMLSessionKHR                  session,
    const VkAllocationCallbacks*    pAllocator)
{
    (void)device;
    if (session == VK_NULL_HANDLE)
        return;

    VkMLSessionKHR_T *s = (VkMLSessionKHR_T *)(uintptr_t)session;
    vk_ml_free(pAllocator, s);
}
