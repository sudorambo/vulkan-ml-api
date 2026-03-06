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

    if (pCreateInfo->scratchMemory != VK_NULL_HANDLE) {
        if (pCreateInfo->scratchMemorySize == 0)
            return VK_ERROR_INITIALIZATION_FAILED;
        if (pCreateInfo->scratchMemoryOffset % _Alignof(max_align_t) != 0)
            return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkMLSessionKHR_T *session = (VkMLSessionKHR_T *)vk_ml_alloc(pAllocator,
        sizeof(VkMLSessionKHR_T));
    if (!session)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    memset(session, 0, sizeof(*session));

    VkMLGraphKHR_T *g = (VkMLGraphKHR_T *)(uintptr_t)pCreateInfo->graph;

    session->graph = pCreateInfo->graph;
    session->autoAllocated = VK_FALSE;

    if (pCreateInfo->scratchMemory != VK_NULL_HANDLE) {
        session->scratchMemory = pCreateInfo->scratchMemory;
        session->scratchMemoryOffset = pCreateInfo->scratchMemoryOffset;
        session->scratchMemorySize = pCreateInfo->scratchMemorySize;
    } else {
        /* Auto-allocate scratch memory (reference implementation) */
        VkDeviceSize needed = g->scratchMemorySize;
        if (needed > 0) {
            void *scratch = vk_ml_alloc(pAllocator, (size_t)needed);
            if (!scratch) {
                vk_ml_free(pAllocator, session);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            session->scratchMemory = (VkDeviceMemory)(uintptr_t)scratch;
            session->scratchMemoryOffset = 0;
            session->scratchMemorySize = needed;
            session->autoAllocated = VK_TRUE;
        }
    }

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
    if (s->autoAllocated && s->scratchMemory != VK_NULL_HANDLE)
        vk_ml_free(pAllocator, (void *)(uintptr_t)s->scratchMemory);
    vk_ml_free(pAllocator, s);
}
