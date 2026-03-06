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
    dst->pNext = NULL;
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
/* Node deep-copy helpers (C1 remediation)                            */
/* ------------------------------------------------------------------ */

static size_t op_desc_size_by_stype(VkStructureType sType)
{
    switch ((int)sType) {
    case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR:
        return sizeof(VkMLPrimitiveDescConvolutionKHR);
    case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR:
        return sizeof(VkMLPrimitiveDescGemmKHR);
    case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR:
        return sizeof(VkMLPrimitiveDescPoolingKHR);
    case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR:
        return sizeof(VkMLPrimitiveDescActivationKHR);
    case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR:
        return sizeof(VkMLPrimitiveDescNormalizationKHR);
    case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR:
        return sizeof(VkMLPrimitiveDescElementwiseKHR);
    default:
        return 0;
    }
}

static void *deep_copy_op_desc(const void *pDesc,
                               const VkAllocationCallbacks *pAllocator)
{
    if (!pDesc)
        return NULL;
    const VkBaseInStructure *base = (const VkBaseInStructure *)pDesc;
    size_t sz = op_desc_size_by_stype(base->sType);
    if (sz == 0)
        return NULL;
    void *copy = vk_ml_alloc(pAllocator, sz);
    if (!copy)
        return NULL;
    memcpy(copy, pDesc, sz);
    ((VkBaseOutStructure *)copy)->pNext = NULL;
    return copy;
}

static void free_binding_deep_data(VkMLTensorBindingKHR *binding,
                                   const VkAllocationCallbacks *pAllocator)
{
    if (!binding)
        return;
    if (binding->pTensorDescription) {
        VkTensorDescriptionKHR *td =
            (VkTensorDescriptionKHR *)binding->pTensorDescription;
        free_tensor_desc_arrays(td, pAllocator);
        vk_ml_free(pAllocator, td);
        binding->pTensorDescription = NULL;
    }
}

static VkMLTensorBindingKHR *deep_copy_bindings(
    const VkMLTensorBindingKHR *pBindings,
    uint32_t count,
    const VkAllocationCallbacks *pAllocator)
{
    if (!pBindings || count == 0)
        return NULL;
    VkMLTensorBindingKHR *dst = (VkMLTensorBindingKHR *)vk_ml_alloc(
        pAllocator, count * sizeof(VkMLTensorBindingKHR));
    if (!dst)
        return NULL;

    for (uint32_t i = 0; i < count; i++) {
        dst[i] = pBindings[i];
        dst[i].pNext = NULL;
        dst[i].pTensorDescription = NULL;

        if (pBindings[i].pTensorDescription) {
            VkTensorDescriptionKHR *td = deep_copy_tensor_desc(
                pBindings[i].pTensorDescription, pAllocator);
            if (!td) {
                for (uint32_t j = 0; j < i; j++)
                    free_binding_deep_data(&dst[j], pAllocator);
                vk_ml_free(pAllocator, dst);
                return NULL;
            }
            dst[i].pTensorDescription = td;
        }
    }
    return dst;
}

static char *deep_copy_string(const char *str,
                              const VkAllocationCallbacks *pAllocator)
{
    if (!str)
        return NULL;
    size_t len = strlen(str) + 1;
    char *copy = (char *)vk_ml_alloc(pAllocator, len);
    if (!copy)
        return NULL;
    memcpy(copy, str, len);
    return copy;
}

static void free_node_deep_data(VkMLGraphNodeCreateInfoKHR *node,
                                const VkAllocationCallbacks *pAllocator)
{
    if (!node)
        return;
    vk_ml_free(pAllocator, (void *)node->pOperationDesc);
    node->pOperationDesc = NULL;

    if (node->pInputs) {
        for (uint32_t i = 0; i < node->inputCount; i++)
            free_binding_deep_data(
                (VkMLTensorBindingKHR *)&node->pInputs[i], pAllocator);
        vk_ml_free(pAllocator, (void *)node->pInputs);
        node->pInputs = NULL;
    }
    if (node->pOutputs) {
        for (uint32_t i = 0; i < node->outputCount; i++)
            free_binding_deep_data(
                (VkMLTensorBindingKHR *)&node->pOutputs[i], pAllocator);
        vk_ml_free(pAllocator, (void *)node->pOutputs);
        node->pOutputs = NULL;
    }
    vk_ml_free(pAllocator, (void *)node->pNodeName);
    node->pNodeName = NULL;
}

/* ------------------------------------------------------------------ */
/* Graph cleanup helper (goto pattern)                                */
/* ------------------------------------------------------------------ */

static void free_graph_internals(VkMLGraphKHR_T *g,
                                 const VkAllocationCallbacks *pAllocator)
{
    if (!g)
        return;
    if (g->nodes) {
        for (uint32_t i = 0; i < g->nodeCount; i++)
            free_node_deep_data(&g->nodes[i], pAllocator);
        vk_ml_free(pAllocator, g->nodes);
    }
    if (g->externalInputDescs) {
        for (uint32_t i = 0; i < g->externalInputCount; i++)
            free_tensor_desc_arrays(&g->externalInputDescs[i], pAllocator);
        vk_ml_free(pAllocator, g->externalInputDescs);
    }
    if (g->externalOutputDescs) {
        for (uint32_t i = 0; i < g->externalOutputCount; i++)
            free_tensor_desc_arrays(&g->externalOutputDescs[i], pAllocator);
        vk_ml_free(pAllocator, g->externalOutputDescs);
    }
    if (g->constantWeightDescs) {
        for (uint32_t i = 0; i < g->constantWeightCount; i++)
            free_tensor_desc_arrays(&g->constantWeightDescs[i], pAllocator);
        vk_ml_free(pAllocator, g->constantWeightDescs);
    }
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

    memset(graph, 0, sizeof(VkMLGraphKHR_T));
    graph->nodeCount = pCreateInfo->nodeCount;
    graph->externalInputCount = pCreateInfo->externalInputCount;
    graph->externalOutputCount = pCreateInfo->externalOutputCount;
    graph->constantWeightCount = pCreateInfo->constantWeightCount;

    VkResult result = VK_SUCCESS;

    /* Deep-copy nodes with all their pointer members */
    if (pCreateInfo->nodeCount > 0 && pCreateInfo->pNodes) {
        graph->nodes = (VkMLGraphNodeCreateInfoKHR *)vk_ml_alloc(pAllocator,
            pCreateInfo->nodeCount * sizeof(VkMLGraphNodeCreateInfoKHR));
        if (!graph->nodes) { result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup; }
        memset(graph->nodes, 0,
            pCreateInfo->nodeCount * sizeof(VkMLGraphNodeCreateInfoKHR));

        for (uint32_t i = 0; i < pCreateInfo->nodeCount; i++) {
            const VkMLGraphNodeCreateInfoKHR *src = &pCreateInfo->pNodes[i];
            VkMLGraphNodeCreateInfoKHR *dst = &graph->nodes[i];

            *dst = *src;
            dst->pNext = NULL;
            dst->pOperationDesc = NULL;
            dst->pInputs = NULL;
            dst->pOutputs = NULL;
            dst->pNodeName = NULL;

            if (src->pOperationDesc) {
                dst->pOperationDesc = deep_copy_op_desc(src->pOperationDesc,
                                                        pAllocator);
                if (!dst->pOperationDesc) {
                    result = VK_ERROR_OUT_OF_HOST_MEMORY;
                    goto cleanup;
                }
            }
            if (src->inputCount > 0 && src->pInputs) {
                dst->pInputs = deep_copy_bindings(src->pInputs,
                                                  src->inputCount, pAllocator);
                if (!dst->pInputs) {
                    result = VK_ERROR_OUT_OF_HOST_MEMORY;
                    goto cleanup;
                }
            }
            if (src->outputCount > 0 && src->pOutputs) {
                dst->pOutputs = deep_copy_bindings(src->pOutputs,
                                                   src->outputCount, pAllocator);
                if (!dst->pOutputs) {
                    result = VK_ERROR_OUT_OF_HOST_MEMORY;
                    goto cleanup;
                }
            }
            if (src->pNodeName) {
                dst->pNodeName = deep_copy_string(src->pNodeName, pAllocator);
                if (!dst->pNodeName) {
                    result = VK_ERROR_OUT_OF_HOST_MEMORY;
                    goto cleanup;
                }
            }
        }
    }

    /* Deep-copy external input descriptions */
    if (pCreateInfo->externalInputCount > 0 &&
        pCreateInfo->pExternalInputDescriptions) {
        graph->externalInputDescs = (VkTensorDescriptionKHR *)vk_ml_alloc(
            pAllocator,
            pCreateInfo->externalInputCount * sizeof(VkTensorDescriptionKHR));
        if (!graph->externalInputDescs) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup;
        }
        memset(graph->externalInputDescs, 0,
            pCreateInfo->externalInputCount * sizeof(VkTensorDescriptionKHR));
        for (uint32_t i = 0; i < pCreateInfo->externalInputCount; i++) {
            VkTensorDescriptionKHR *copy = deep_copy_tensor_desc(
                &pCreateInfo->pExternalInputDescriptions[i], pAllocator);
            if (!copy) { result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup; }
            graph->externalInputDescs[i] = *copy;
            vk_ml_free(pAllocator, copy);
        }
    }

    /* Deep-copy external output descriptions */
    if (pCreateInfo->externalOutputCount > 0 &&
        pCreateInfo->pExternalOutputDescriptions) {
        graph->externalOutputDescs = (VkTensorDescriptionKHR *)vk_ml_alloc(
            pAllocator,
            pCreateInfo->externalOutputCount * sizeof(VkTensorDescriptionKHR));
        if (!graph->externalOutputDescs) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup;
        }
        memset(graph->externalOutputDescs, 0,
            pCreateInfo->externalOutputCount * sizeof(VkTensorDescriptionKHR));
        for (uint32_t i = 0; i < pCreateInfo->externalOutputCount; i++) {
            VkTensorDescriptionKHR *copy = deep_copy_tensor_desc(
                &pCreateInfo->pExternalOutputDescriptions[i], pAllocator);
            if (!copy) { result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup; }
            graph->externalOutputDescs[i] = *copy;
            vk_ml_free(pAllocator, copy);
        }
    }

    /* Deep-copy constant weight descriptions */
    if (pCreateInfo->constantWeightCount > 0 &&
        pCreateInfo->pConstantWeightDescriptions) {
        graph->constantWeightDescs = (VkTensorDescriptionKHR *)vk_ml_alloc(
            pAllocator,
            pCreateInfo->constantWeightCount * sizeof(VkTensorDescriptionKHR));
        if (!graph->constantWeightDescs) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup;
        }
        memset(graph->constantWeightDescs, 0,
            pCreateInfo->constantWeightCount * sizeof(VkTensorDescriptionKHR));
        for (uint32_t i = 0; i < pCreateInfo->constantWeightCount; i++) {
            VkTensorDescriptionKHR *copy = deep_copy_tensor_desc(
                &pCreateInfo->pConstantWeightDescriptions[i], pAllocator);
            if (!copy) { result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup; }
            graph->constantWeightDescs[i] = *copy;
            vk_ml_free(pAllocator, copy);
        }
    }

    /* Scratch size: sum of all tensor sizes * 2 */
    {
        VkDeviceSize totalSize = 0;
        for (uint32_t i = 0; i < graph->externalInputCount; i++)
            totalSize += tensor_desc_size(&graph->externalInputDescs[i]);
        for (uint32_t i = 0; i < graph->externalOutputCount; i++)
            totalSize += tensor_desc_size(&graph->externalOutputDescs[i]);
        for (uint32_t i = 0; i < graph->constantWeightCount; i++)
            totalSize += tensor_desc_size(&graph->constantWeightDescs[i]);
        graph->scratchMemorySize = totalSize * 2;
    }

    *pGraph = (VkMLGraphKHR)(uintptr_t)graph;
    return VK_SUCCESS;

cleanup:
    free_graph_internals(graph, pAllocator);
    vk_ml_free(pAllocator, graph);
    return result;
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
    free_graph_internals(g, pAllocator);
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
