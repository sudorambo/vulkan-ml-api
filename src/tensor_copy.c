/**
 * @file tensor_copy.c
 * @brief VK_KHR_ml_primitives tensor copy command implementation.
 *
 * Stub implementation: validates basic parameters and returns.
 * Actual GPU copy is IHV-specific; reference impl records intent.
 */

#include "internal.h"

/* ------------------------------------------------------------------ */
/* Tensor copy command                                                */
/* ------------------------------------------------------------------ */

VKAPI_ATTR void VKAPI_CALL vkCmdCopyTensorKHR(
    VkCommandBuffer                 commandBuffer,
    const VkCopyTensorInfoKHR*      pCopyInfo)
{
    (void)commandBuffer;
    (void)pCopyInfo;

    if (!pCopyInfo)
        return;

    /* Basic parameter validation */
    if (pCopyInfo->srcTensor == VK_NULL_HANDLE || pCopyInfo->dstTensor == VK_NULL_HANDLE)
        return;
    if (pCopyInfo->srcTensor == pCopyInfo->dstTensor)
        return;
    if (pCopyInfo->regionCount > 0 && !pCopyInfo->pRegions)
        return;

    /* Stub: actual GPU copy is IHV-specific; reference impl records intent */
}
