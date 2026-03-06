/**
 * @file ml_dispatch.c
 * @brief VK_KHR_ml_primitives ML graph dispatch command implementation.
 *
 * Stub implementation: validates non-null parameters and returns.
 * Actual GPU execution is IHV-specific.
 */

#include "internal.h"

/* ------------------------------------------------------------------ */
/* ML graph dispatch command                                         */
/* ------------------------------------------------------------------ */

VKAPI_ATTR void VKAPI_CALL vkCmdDispatchMLGraphKHR(
    VkCommandBuffer                     commandBuffer,
    const VkMLGraphDispatchInfoKHR*     pDispatchInfo)
{
    (void)commandBuffer;
    (void)pDispatchInfo;

    if (!pDispatchInfo)
        return;

    /* Basic parameter validation */
    if (pDispatchInfo->session == VK_NULL_HANDLE)
        return;

    /* Stub: actual GPU execution is IHV-specific; reference impl records intent */
}
