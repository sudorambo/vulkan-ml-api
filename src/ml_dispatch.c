/**
 * @file ml_dispatch.c
 * @brief VK_KHR_ml_primitives ML graph dispatch command implementation.
 *
 * Stub implementation: validates non-null parameters and returns.
 * Actual GPU execution is IHV-specific.
 */

#include <vulkan/vulkan_ml_primitives.h>

/* ------------------------------------------------------------------ */
/* ML graph dispatch command                                         */
/* ------------------------------------------------------------------ */

VKAPI_ATTR void VKAPI_CALL vkCmdDispatchMLGraphKHR(
    VkCommandBuffer                     commandBuffer,
    const VkMLGraphDispatchInfoKHR*     pDispatchInfo)
{
    if (!commandBuffer || !pDispatchInfo)
        return;
    if ((uint32_t)pDispatchInfo->sType != VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR)
        return;

    if (pDispatchInfo->session == VK_NULL_HANDLE)
        return;
    if (pDispatchInfo->inputTensorCount == 0 || !pDispatchInfo->pInputTensors)
        return;
    if (pDispatchInfo->outputTensorCount == 0 || !pDispatchInfo->pOutputTensors)
        return;
    if (pDispatchInfo->weightTensorCount > 0 && !pDispatchInfo->pWeightTensors)
        return;

    /* Reference ICD: dispatch intent recorded; actual GPU execution is IHV-specific */
}
