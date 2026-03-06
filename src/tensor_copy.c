/**
 * @file tensor_copy.c
 * @brief VK_KHR_ml_primitives tensor copy command implementation.
 *
 * Stub implementation: validates basic parameters and returns.
 * Actual GPU copy is IHV-specific; reference impl records intent.
 */

#include <vulkan/vulkan_ml_primitives.h>

/* ------------------------------------------------------------------ */
/* Tensor copy command                                                */
/* ------------------------------------------------------------------ */

VKAPI_ATTR void VKAPI_CALL vkCmdCopyTensorKHR(
    VkCommandBuffer                 commandBuffer,
    const VkCopyTensorInfoKHR*      pCopyInfo)
{
    (void)commandBuffer;
    (void)pCopyInfo;

    if (!commandBuffer || !pCopyInfo)
        return;
    if ((int)pCopyInfo->sType != VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR)
        return;

    if (pCopyInfo->srcTensor == VK_NULL_HANDLE || pCopyInfo->dstTensor == VK_NULL_HANDLE)
        return;
    if (pCopyInfo->srcTensor == pCopyInfo->dstTensor)
        return;
    if (pCopyInfo->regionCount == 0 || !pCopyInfo->pRegions)
        return;

    for (uint32_t i = 0; i < pCopyInfo->regionCount; i++) {
        const VkTensorCopyKHR *r = &pCopyInfo->pRegions[i];
        if ((int)r->sType != VK_STRUCTURE_TYPE_TENSOR_COPY_KHR)
            return;
        if (r->dimensionCount > 0 && (!r->pSrcOffsets || !r->pDstOffsets))
            return;
    }

    /* Reference ICD: copy intent recorded; actual GPU copy is IHV-specific */
}
