/**
 * @file tensor.c
 * @brief VK_KHR_ml_primitives tensor lifecycle and memory implementation.
 */

#include "internal.h"

/* ------------------------------------------------------------------ */
/* Tensor creation and destruction                                    */
/* ------------------------------------------------------------------ */

VKAPI_ATTR VkResult VKAPI_CALL vkCreateTensorKHR(
    VkDevice                        device,
    const VkTensorCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*    pAllocator,
    VkTensorKHR*                    pTensor)
{
    (void)device;
    if (!pCreateInfo || !pCreateInfo->pDescription || !pTensor)
        return VK_ERROR_UNKNOWN;
    if ((uint32_t)pCreateInfo->sType != VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR)
        return VK_ERROR_UNKNOWN;

    const VkTensorDescriptionKHR* desc = pCreateInfo->pDescription;
    const uint32_t dimCount = desc->dimensionCount;

    VkTensorKHR_T* tensor = (VkTensorKHR_T*)vk_ml_alloc(pAllocator, sizeof(VkTensorKHR_T));
    if (!tensor)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    tensor->description = *desc;
    tensor->sharingMode = pCreateInfo->sharingMode;
    tensor->queueFamilyIndexCount = pCreateInfo->queueFamilyIndexCount;
    tensor->boundMemory = VK_NULL_HANDLE;
    tensor->memoryOffset = 0;
    tensor->memoryBound = VK_FALSE;

    tensor->dimensions = NULL;
    tensor->strides = NULL;
    tensor->queueFamilyIndices = NULL;

    if (dimCount > 0 && desc->pDimensions) {
        tensor->dimensions = (uint32_t*)vk_ml_alloc(pAllocator, dimCount * sizeof(uint32_t));
        if (!tensor->dimensions) {
            vk_ml_free(pAllocator, tensor);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        memcpy(tensor->dimensions, desc->pDimensions, dimCount * sizeof(uint32_t));
    }

    if (dimCount > 0 && desc->pStrides) {
        tensor->strides = (VkDeviceSize*)vk_ml_alloc(pAllocator, dimCount * sizeof(VkDeviceSize));
        if (!tensor->strides) {
            vk_ml_free(pAllocator, tensor->dimensions);
            vk_ml_free(pAllocator, tensor);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        memcpy(tensor->strides, desc->pStrides, dimCount * sizeof(VkDeviceSize));
    }

    tensor->description.pDimensions = tensor->dimensions;
    tensor->description.pStrides = tensor->strides;
    tensor->description.pNext = NULL;

    if (pCreateInfo->queueFamilyIndexCount > 0 && pCreateInfo->pQueueFamilyIndices) {
        tensor->queueFamilyIndices = (uint32_t*)vk_ml_alloc(pAllocator,
            pCreateInfo->queueFamilyIndexCount * sizeof(uint32_t));
        if (!tensor->queueFamilyIndices) {
            vk_ml_free(pAllocator, tensor->strides);
            vk_ml_free(pAllocator, tensor->dimensions);
            vk_ml_free(pAllocator, tensor);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        memcpy(tensor->queueFamilyIndices, pCreateInfo->pQueueFamilyIndices,
            pCreateInfo->queueFamilyIndexCount * sizeof(uint32_t));
    }

    *pTensor = (VkTensorKHR)(uintptr_t)tensor;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyTensorKHR(
    VkDevice                        device,
    VkTensorKHR                     tensor,
    const VkAllocationCallbacks*    pAllocator)
{
    (void)device;
    if (tensor == VK_NULL_HANDLE)
        return;

    VkTensorKHR_T* t = (VkTensorKHR_T*)(uintptr_t)tensor;
    vk_ml_free(pAllocator, t->dimensions);
    vk_ml_free(pAllocator, t->strides);
    vk_ml_free(pAllocator, t->queueFamilyIndices);
    vk_ml_free(pAllocator, t);
}

/* ------------------------------------------------------------------ */
/* Tensor memory requirements and binding                             */
/* ------------------------------------------------------------------ */

VKAPI_ATTR void VKAPI_CALL vkGetTensorMemoryRequirementsKHR(
    VkDevice                                    device,
    const VkTensorMemoryRequirementsInfoKHR*   pInfo,
    VkMemoryRequirements2*                      pMemoryRequirements)
{
    (void)device;
    if (!pInfo || !pMemoryRequirements || pInfo->tensor == VK_NULL_HANDLE)
        return;

    VkTensorKHR_T* t = (VkTensorKHR_T*)(uintptr_t)pInfo->tensor;
    const VkTensorDescriptionKHR* desc = &t->description;

    VkDeviceSize elementCount = 1;
    const uint32_t* dims = desc->pDimensions;
    if (dims && desc->dimensionCount > 0) {
        for (uint32_t i = 0; i < desc->dimensionCount; i++)
            elementCount *= (VkDeviceSize)dims[i];
    }

    uint32_t elemSize = vk_ml_format_element_size(desc->format);
    VkDeviceSize size = elementCount * elemSize;

    pMemoryRequirements->sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    pMemoryRequirements->pNext = NULL;
    pMemoryRequirements->memoryRequirements.size = size;
    pMemoryRequirements->memoryRequirements.alignment = VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN;
    pMemoryRequirements->memoryRequirements.memoryTypeBits = 0xFFFFFFFFu;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindTensorMemoryKHR(
    VkDevice                              device,
    uint32_t                              bindInfoCount,
    const VkBindTensorMemoryInfoKHR*      pBindInfos)
{
    (void)device;
    if (!pBindInfos)
        return VK_ERROR_UNKNOWN;

    for (uint32_t i = 0; i < bindInfoCount; i++) {
        const VkBindTensorMemoryInfoKHR* info = &pBindInfos[i];
        if (info->tensor == VK_NULL_HANDLE)
            return VK_ERROR_UNKNOWN;

        VkTensorKHR_T* t = (VkTensorKHR_T*)(uintptr_t)info->tensor;
        if (t->memoryBound)
            return VK_ERROR_UNKNOWN;
        if (VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN > 0 &&
            info->memoryOffset % VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN != 0)
            return VK_ERROR_UNKNOWN;
        t->boundMemory = info->memory;
        t->memoryOffset = info->memoryOffset;
        t->memoryBound = VK_TRUE;
    }
    return VK_SUCCESS;
}
