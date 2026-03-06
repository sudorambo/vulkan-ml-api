/**
 * @file tensor_view.c
 * @brief VK_KHR_ml_primitives tensor view implementation.
 */

#include "internal.h"

/* ------------------------------------------------------------------ */
/* Tensor view creation and destruction                               */
/* ------------------------------------------------------------------ */

VKAPI_ATTR VkResult VKAPI_CALL vkCreateTensorViewKHR(
    VkDevice                            device,
    const VkTensorViewCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*        pAllocator,
    VkTensorViewKHR*                    pView)
{
    (void)device;
    if (!pCreateInfo || !pView)
        return VK_ERROR_UNKNOWN;
    if (pCreateInfo->tensor == VK_NULL_HANDLE)
        return VK_ERROR_UNKNOWN;

    VkTensorViewKHR_T* view = (VkTensorViewKHR_T*)vk_ml_alloc(pAllocator, sizeof(VkTensorViewKHR_T));
    if (!view)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    view->tensor = pCreateInfo->tensor;
    view->format = pCreateInfo->format;
    view->dimensionCount = pCreateInfo->dimensionCount;
    view->dimensionOffsets = NULL;
    view->dimensionSizes = NULL;

    if (pCreateInfo->dimensionCount > 0) {
        if (pCreateInfo->pDimensionOffsets) {
            view->dimensionOffsets = (uint32_t*)vk_ml_alloc(pAllocator,
                pCreateInfo->dimensionCount * sizeof(uint32_t));
            if (!view->dimensionOffsets) {
                vk_ml_free(pAllocator, view);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            memcpy(view->dimensionOffsets, pCreateInfo->pDimensionOffsets,
                pCreateInfo->dimensionCount * sizeof(uint32_t));
        }
        if (pCreateInfo->pDimensionSizes) {
            view->dimensionSizes = (uint32_t*)vk_ml_alloc(pAllocator,
                pCreateInfo->dimensionCount * sizeof(uint32_t));
            if (!view->dimensionSizes) {
                vk_ml_free(pAllocator, view->dimensionOffsets);
                vk_ml_free(pAllocator, view);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            memcpy(view->dimensionSizes, pCreateInfo->pDimensionSizes,
                pCreateInfo->dimensionCount * sizeof(uint32_t));
        }
    }

    *pView = (VkTensorViewKHR)(uintptr_t)view;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyTensorViewKHR(
    VkDevice                        device,
    VkTensorViewKHR                 view,
    const VkAllocationCallbacks*    pAllocator)
{
    (void)device;
    if (view == VK_NULL_HANDLE)
        return;

    VkTensorViewKHR_T* v = (VkTensorViewKHR_T*)(uintptr_t)view;
    vk_ml_free(pAllocator, v->dimensionOffsets);
    vk_ml_free(pAllocator, v->dimensionSizes);
    vk_ml_free(pAllocator, v);
}
