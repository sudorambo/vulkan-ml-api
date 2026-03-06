/**
 * @file dispatch_validation.c
 * @brief ML graph dispatch validation for VK_KHR_ml_primitives.
 */

#include "../validation/vk_ml_validation.h"

#include <stddef.h>

VkBool32 vk_ml_validate_dispatch(
    const VkMLGraphDispatchInfoKHR *pDispatchInfo)
{
    if (!pDispatchInfo)
        return VK_FALSE;

    /* VUID_DISPATCH_SESSION */
    if (pDispatchInfo->session == VK_NULL_HANDLE)
        return VK_FALSE;

    /* VUID_DISPATCH_INPUT_COUNT - inputTensorCount must match graph's externalInputCount */
    /* VUID_DISPATCH_OUTPUT_COUNT - outputTensorCount must match graph's externalOutputCount */
    if (pDispatchInfo->inputTensorCount == 0 || pDispatchInfo->outputTensorCount == 0)
        return VK_FALSE;

    if (!pDispatchInfo->pInputTensors)
        return VK_FALSE;
    if (!pDispatchInfo->pOutputTensors)
        return VK_FALSE;
    if (pDispatchInfo->weightTensorCount > 0 && !pDispatchInfo->pWeightTensors)
        return VK_FALSE;

    /* VUID_DISPATCH_WEIGHT_COUNT - weightTensorCount must match graph's constantWeightCount */
    /* Stub: full count matching requires graph context */

    return VK_TRUE;
}
