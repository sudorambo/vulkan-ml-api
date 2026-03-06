/**
 * @file tensor_validation.c
 * @brief Tensor-related validation for VK_KHR_ml_primitives.
 */

#include "vk_ml_validation.h"
#include "internal.h"

#include <stddef.h>

VkBool32 vk_ml_validate_tensor_create(
    const VkTensorCreateInfoKHR *pCreateInfo,
    const VkPhysicalDeviceMLFeaturesKHR *features,
    const VkPhysicalDeviceMLPropertiesKHR *props)
{
    if (!pCreateInfo || !features || !props)
        return VK_FALSE;
    if ((int)pCreateInfo->sType != VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR)
        return VK_FALSE;

    /* VUID_TENSOR_OBJECTS_FEATURE */
    if (!features->tensorObjects)
        return VK_FALSE;

    /* VUID_TENSOR_CREATE_DESC */
    const VkTensorDescriptionKHR *desc = pCreateInfo->pDescription;
    if (!desc)
        return VK_FALSE;

    /* VUID_TENSOR_CREATE_SHARING_MODE */
    if (pCreateInfo->sharingMode != VK_SHARING_MODE_EXCLUSIVE &&
        pCreateInfo->sharingMode != VK_SHARING_MODE_CONCURRENT)
        return VK_FALSE;

    /* VUID_TENSOR_CREATE_SHARING_INDICES */
    if (pCreateInfo->sharingMode == VK_SHARING_MODE_CONCURRENT) {
        if (pCreateInfo->queueFamilyIndexCount < 2 || !pCreateInfo->pQueueFamilyIndices)
            return VK_FALSE;
    }

    /* VUID_TENSOR_DESC_DIM_COUNT */
    if (desc->dimensionCount == 0 || desc->dimensionCount > props->maxTensorDimensions)
        return VK_FALSE;

    if (!desc->pDimensions)
        return VK_FALSE;

    uint64_t product = 1;
    for (uint32_t i = 0; i < desc->dimensionCount; i++) {
        /* VUID_TENSOR_DESC_DIM_VALUES */
        if (desc->pDimensions[i] == 0 || desc->pDimensions[i] > props->maxTensorDimensionSize)
            return VK_FALSE;
        if (product > props->maxTensorElements / desc->pDimensions[i])
            return VK_FALSE;
        product *= desc->pDimensions[i];
    }

    /* VUID_TENSOR_DESC_DIM_PRODUCT */
    if (product > props->maxTensorElements)
        return VK_FALSE;

    /* VUID_TENSOR_DESC_STRIDES_OPTIMAL */
    if (desc->tiling == VK_TENSOR_TILING_OPTIMAL_KHR && desc->pStrides != NULL)
        return VK_FALSE;

    /* VUID_TENSOR_DESC_FORMAT */
    uint32_t elemSize = vk_ml_format_element_size(desc->format);
    if (elemSize == 0)
        return VK_FALSE;

    /* VUID_TENSOR_DESC_STRIDE_ALIGN */
    if (desc->pStrides != NULL) {
        for (uint32_t i = 0; i < desc->dimensionCount; i++) {
            if (desc->pStrides[i] % elemSize != 0)
                return VK_FALSE;
        }
    }

    /* VUID_TENSOR_USAGE */
    if (desc->usage == 0)
        return VK_FALSE;
    const VkFlags validUsageMask = (VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_KHR << 1) - 1;
    if (desc->usage & ~validUsageMask)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_tensor_view_create(
    const VkTensorViewCreateInfoKHR *pCreateInfo,
    const struct VkTensorKHR_T *tensor)
{
    if (!pCreateInfo || !tensor)
        return VK_FALSE;
    if ((int)pCreateInfo->sType != VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR)
        return VK_FALSE;

    /* VUID_TENSOR_VIEW_MEMORY_BOUND */
    if (!tensor->memoryBound)
        return VK_FALSE;

    /* VUID_TENSOR_VIEW_FORMAT */
    uint32_t tensorElemSize = vk_ml_format_element_size(tensor->description.format);
    uint32_t viewElemSize = vk_ml_format_element_size(pCreateInfo->format);
    if (tensorElemSize == 0 || viewElemSize == 0 || tensorElemSize != viewElemSize)
        return VK_FALSE;

    /* VUID_TENSOR_VIEW_DIM_COUNT */
    if (pCreateInfo->dimensionCount != tensor->description.dimensionCount)
        return VK_FALSE;

    if (!tensor->dimensions || !pCreateInfo->pDimensionOffsets || !pCreateInfo->pDimensionSizes)
        return VK_FALSE;

    /* VUID_TENSOR_VIEW_RANGE */
    for (uint32_t i = 0; i < pCreateInfo->dimensionCount; i++) {
        if (pCreateInfo->pDimensionOffsets[i] + pCreateInfo->pDimensionSizes[i] > tensor->dimensions[i])
            return VK_FALSE;
    }

    return VK_TRUE;
}

VkBool32 vk_ml_validate_tensor_bind(
    const VkBindTensorMemoryInfoKHR *pBindInfo,
    const struct VkTensorKHR_T *tensor,
    const VkPhysicalDeviceMLPropertiesKHR *props)
{
    if (!pBindInfo || !tensor || !props)
        return VK_FALSE;
    if ((int)pBindInfo->sType != VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR)
        return VK_FALSE;

    /* VUID_BIND_TENSOR_ALREADY_BOUND */
    if (tensor->memoryBound)
        return VK_FALSE;

    /* VUID_BIND_TENSOR_MEM_TYPE */
    if (pBindInfo->memory == VK_NULL_HANDLE)
        return VK_FALSE;

    /* VUID_BIND_TENSOR_ALIGNMENT */
    if (props->minTensorMemoryAlignment > 0 &&
        pBindInfo->memoryOffset % props->minTensorMemoryAlignment != 0)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_tensor_copy(
    const VkCopyTensorInfoKHR *pCopyInfo)
{
    if (!pCopyInfo)
        return VK_FALSE;
    if ((int)pCopyInfo->sType != VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR)
        return VK_FALSE;

    /* VUID_COPY_TENSOR_SAME */
    if (pCopyInfo->srcTensor == pCopyInfo->dstTensor)
        return VK_FALSE;

    /* VUID_COPY_TENSOR_REGION_COUNT */
    if (pCopyInfo->regionCount == 0)
        return VK_FALSE;

    if (!pCopyInfo->pRegions)
        return VK_FALSE;

    for (uint32_t i = 0; i < pCopyInfo->regionCount; i++) {
        const VkTensorCopyKHR *region = &pCopyInfo->pRegions[i];
        if ((int)region->sType != VK_STRUCTURE_TYPE_TENSOR_COPY_KHR)
            return VK_FALSE;
        /* VUID_COPY_TENSOR_SRC_OFFSETS */
        if (region->dimensionCount > 0 && !region->pSrcOffsets)
            return VK_FALSE;
        /* VUID_COPY_TENSOR_DST_OFFSETS */
        if (region->dimensionCount > 0 && !region->pDstOffsets)
            return VK_FALSE;
    }

    return VK_TRUE;
}
