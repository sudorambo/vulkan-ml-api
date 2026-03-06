/**
 * @file ml_graph.c
 * @brief VK_KHR_ml_primitives ML graph lifecycle implementation.
 */

#include "internal.h"

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static VkDeviceSize tensor_desc_size(const VkTensorDescriptionKHR *desc)
{
    if (!desc || !desc->pDimensions || desc->dimensionCount == 0)
        return 0;
    VkDeviceSize count = 1;
    for (uint32_t i = 0; i < desc->dimensionCount; i++)
        count *= (VkDeviceSize)desc->pDimensions[i];
    return count * vk_ml_format_element_size(desc->format);
}

static VkTensorDescriptionKHR *deep_copy_tensor_desc(
    const VkTensorDescriptionKHR *src,
    const VkAllocationCallbacks *pAllocator)
{
    if (!src)
        return NULL;
    VkTensorDescriptionKHR *dst = (VkTensorDescriptionKHR *)vk_ml_alloc(
        pAllocator, sizeof(VkTensorDescriptionKHR));
    if (!dst)
        return NULL;
    *dst = *src;
    dst->pDimensions = NULL;
    dst->pStrides = NULL;

    if (src->dimensionCount > 0 && src->pDimensions) {
        dst->pDimensions = (const uint32_t *)vk_ml_alloc(pAllocator,
            src->dimensionCount * sizeof(uint32_t));
        if (!dst->pDimensions) {
            vk_ml_free(pAllocator, dst);
            return NULL;
        }
        memcpy((void *)dst->pDimensions, src->pDimensions,
            src->dimensionCount * sizeof(uint32_t));
    }
    if (src->dimensionCount > 0 && src->pStrides) {
        dst->pStrides = (const VkDeviceSize *)vk_ml_alloc(pAllocator,
            src->dimensionCount * sizeof(VkDeviceSize));
        if (!dst->pStrides) {
            vk_ml_free(pAllocator, (void *)dst->pDimensions);
            vk_ml_free(pAllocator, dst);
            return NULL;
        }
        memcpy((void *)dst->pStrides, src->pStrides,
            src->dimensionCount * sizeof(VkDeviceSize));
    }
    return dst;
}

static void free_tensor_desc_arrays(VkTensorDescriptionKHR *desc,
                                    const VkAllocationCallbacks *pAllocator)
{
    if (!desc)
        return;
    vk_ml_free(pAllocator, (void *)desc->pDimensions);
    vk_ml_free(pAllocator, (void *)desc->pStrides);
}

/* ------------------------------------------------------------------ */
/* ML graph creation and destruction                                  */
/* ------------------------------------------------------------------ */

VKAPI_ATTR VkResult VKAPI_CALL vkCreateMLGraphKHR(
    VkDevice                         device,
    const VkMLGraphCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*     pAllocator,
    VkMLGraphKHR*                    pGraph)
{
    (void)device;
    if (!pCreateInfo || !pGraph)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkMLGraphKHR_T *graph = (VkMLGraphKHR_T *)vk_ml_alloc(pAllocator,
        sizeof(VkMLGraphKHR_T));
    if (!graph)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    graph->nodeCount = pCreateInfo->nodeCount;
    graph->nodes = NULL;
    graph->externalInputCount = pCreateInfo->externalInputCount;
    graph->externalInputDescs = NULL;
    graph->externalOutputCount = pCreateInfo->externalOutputCount;
    graph->externalOutputDescs = NULL;
    graph->constantWeightCount = pCreateInfo->constantWeightCount;
    graph->constantWeightDescs = NULL;
    graph->scratchMemorySize = 0;

    if (pCreateInfo->nodeCount > 0 && pCreateInfo->pNodes) {
        graph->nodes = (VkMLGraphNodeCreateInfoKHR *)vk_ml_alloc(pAllocator,
            pCreateInfo->nodeCount * sizeof(VkMLGraphNodeCreateInfoKHR));
        if (!graph->nodes) {
            vk_ml_free(pAllocator, graph);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        memcpy(graph->nodes, pCreateInfo->pNodes,
            pCreateInfo->nodeCount * sizeof(VkMLGraphNodeCreateInfoKHR));
    }

    /* Deep copy external input descriptions */
    if (pCreateInfo->externalInputCount > 0 && pCreateInfo->pExternalInputDescriptions) {
        graph->externalInputDescs = (VkTensorDescriptionKHR *)vk_ml_alloc(
            pAllocator,
            pCreateInfo->externalInputCount * sizeof(VkTensorDescriptionKHR));
        if (!graph->externalInputDescs) {
            vk_ml_free(pAllocator, graph->nodes);
            vk_ml_free(pAllocator, graph);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        for (uint32_t i = 0; i < pCreateInfo->externalInputCount; i++) {
            VkTensorDescriptionKHR *copy = deep_copy_tensor_desc(
                &pCreateInfo->pExternalInputDescriptions[i], pAllocator);
            if (!copy) {
                for (uint32_t j = 0; j < i; j++)
                    free_tensor_desc_arrays(&graph->externalInputDescs[j], pAllocator);
                vk_ml_free(pAllocator, graph->externalInputDescs);
                vk_ml_free(pAllocator, graph->nodes);
                vk_ml_free(pAllocator, graph);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            graph->externalInputDescs[i] = *copy;
            vk_ml_free(pAllocator, copy);
        }
    }

    if (pCreateInfo->externalOutputCount > 0 && pCreateInfo->pExternalOutputDescriptions) {
        graph->externalOutputDescs = (VkTensorDescriptionKHR *)vk_ml_alloc(
            pAllocator,
            pCreateInfo->externalOutputCount * sizeof(VkTensorDescriptionKHR));
        if (!graph->externalOutputDescs) {
            for (uint32_t i = 0; i < graph->externalInputCount; i++)
                free_tensor_desc_arrays(&graph->externalInputDescs[i], pAllocator);
            vk_ml_free(pAllocator, graph->externalInputDescs);
            vk_ml_free(pAllocator, graph->nodes);
            vk_ml_free(pAllocator, graph);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        for (uint32_t i = 0; i < pCreateInfo->externalOutputCount; i++) {
            VkTensorDescriptionKHR *copy = deep_copy_tensor_desc(
                &pCreateInfo->pExternalOutputDescriptions[i], pAllocator);
            if (!copy) {
                for (uint32_t j = 0; j < i; j++)
                    free_tensor_desc_arrays(&graph->externalOutputDescs[j], pAllocator);
                vk_ml_free(pAllocator, graph->externalOutputDescs);
                for (uint32_t j = 0; j < graph->externalInputCount; j++)
                    free_tensor_desc_arrays(&graph->externalInputDescs[j], pAllocator);
                vk_ml_free(pAllocator, graph->externalInputDescs);
                vk_ml_free(pAllocator, graph->nodes);
                vk_ml_free(pAllocator, graph);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            graph->externalOutputDescs[i] = *copy;
            vk_ml_free(pAllocator, copy);
        }
    }

    if (pCreateInfo->constantWeightCount > 0 && pCreateInfo->pConstantWeightDescriptions) {
        graph->constantWeightDescs = (VkTensorDescriptionKHR *)vk_ml_alloc(
            pAllocator,
            pCreateInfo->constantWeightCount * sizeof(VkTensorDescriptionKHR));
        if (!graph->constantWeightDescs) {
            for (uint32_t i = 0; i < graph->externalOutputCount; i++)
                free_tensor_desc_arrays(&graph->externalOutputDescs[i], pAllocator);
            vk_ml_free(pAllocator, graph->externalOutputDescs);
            for (uint32_t i = 0; i < graph->externalInputCount; i++)
                free_tensor_desc_arrays(&graph->externalInputDescs[i], pAllocator);
            vk_ml_free(pAllocator, graph->externalInputDescs);
            vk_ml_free(pAllocator, graph->nodes);
            vk_ml_free(pAllocator, graph);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        for (uint32_t i = 0; i < pCreateInfo->constantWeightCount; i++) {
            VkTensorDescriptionKHR *copy = deep_copy_tensor_desc(
                &pCreateInfo->pConstantWeightDescriptions[i], pAllocator);
            if (!copy) {
                for (uint32_t j = 0; j < i; j++)
                    free_tensor_desc_arrays(&graph->constantWeightDescs[j], pAllocator);
                vk_ml_free(pAllocator, graph->constantWeightDescs);
                for (uint32_t j = 0; j < graph->externalOutputCount; j++)
                    free_tensor_desc_arrays(&graph->externalOutputDescs[j], pAllocator);
                vk_ml_free(pAllocator, graph->externalOutputDescs);
                for (uint32_t j = 0; j < graph->externalInputCount; j++)
                    free_tensor_desc_arrays(&graph->externalInputDescs[j], pAllocator);
                vk_ml_free(pAllocator, graph->externalInputDescs);
                vk_ml_free(pAllocator, graph->nodes);
                vk_ml_free(pAllocator, graph);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            graph->constantWeightDescs[i] = *copy;
            vk_ml_free(pAllocator, copy);
        }
    }

    /* Dummy scratch size: sum of all tensor sizes * 2 */
    VkDeviceSize totalSize = 0;
    for (uint32_t i = 0; i < graph->externalInputCount; i++)
        totalSize += tensor_desc_size(&graph->externalInputDescs[i]);
    for (uint32_t i = 0; i < graph->externalOutputCount; i++)
        totalSize += tensor_desc_size(&graph->externalOutputDescs[i]);
    for (uint32_t i = 0; i < graph->constantWeightCount; i++)
        totalSize += tensor_desc_size(&graph->constantWeightDescs[i]);
    graph->scratchMemorySize = totalSize * 2;

    *pGraph = (VkMLGraphKHR)(uintptr_t)graph;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyMLGraphKHR(
    VkDevice                        device,
    VkMLGraphKHR                    graph,
    const VkAllocationCallbacks*    pAllocator)
{
    (void)device;
    if (graph == VK_NULL_HANDLE)
        return;

    VkMLGraphKHR_T *g = (VkMLGraphKHR_T *)(uintptr_t)graph;
    for (uint32_t i = 0; i < g->externalInputCount; i++)
        free_tensor_desc_arrays(&g->externalInputDescs[i], pAllocator);
    vk_ml_free(pAllocator, g->externalInputDescs);
    for (uint32_t i = 0; i < g->externalOutputCount; i++)
        free_tensor_desc_arrays(&g->externalOutputDescs[i], pAllocator);
    vk_ml_free(pAllocator, g->externalOutputDescs);
    for (uint32_t i = 0; i < g->constantWeightCount; i++)
        free_tensor_desc_arrays(&g->constantWeightDescs[i], pAllocator);
    vk_ml_free(pAllocator, g->constantWeightDescs);
    vk_ml_free(pAllocator, g->nodes);
    vk_ml_free(pAllocator, g);
}

/* ------------------------------------------------------------------ */
/* ML graph memory requirements                                       */
/* ------------------------------------------------------------------ */

VKAPI_ATTR void VKAPI_CALL vkGetMLGraphMemoryRequirementsKHR(
    VkDevice                device,
    VkMLGraphKHR            graph,
    VkMemoryRequirements2*  pMemoryRequirements)
{
    (void)device;
    if (!pMemoryRequirements || graph == VK_NULL_HANDLE)
        return;

    VkMLGraphKHR_T *g = (VkMLGraphKHR_T *)(uintptr_t)graph;

    pMemoryRequirements->sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    pMemoryRequirements->memoryRequirements.size = g->scratchMemorySize;
    pMemoryRequirements->memoryRequirements.alignment = VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN;
    pMemoryRequirements->memoryRequirements.memoryTypeBits = 0xFFFFFFFFu;
}
