/**
 * @file graph_validation.c
 * @brief ML graph and primitive descriptor validation for VK_KHR_ml_primitives.
 */

#include "vk_ml_validation.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static VkBool32 dfs_has_cycle(uint32_t node,
                              const VkMLGraphNodeCreateInfoKHR *nodes,
                              uint32_t nodeCount,
                              uint8_t *color)
{
    color[node] = 1; /* GRAY - in progress */
    for (uint32_t i = 0; i < nodes[node].inputCount; i++) {
        if (!nodes[node].pInputs)
            continue;
        const VkMLTensorBindingKHR *binding = &nodes[node].pInputs[i];
        if (binding->bindingType != VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR)
            continue;
        uint32_t pred = binding->nodeIndex;
        if (pred >= nodeCount)
            return VK_TRUE; /* invalid reference */
        if (color[pred] == 1)
            return VK_TRUE; /* back edge = cycle */
        if (color[pred] == 0 && dfs_has_cycle(pred, nodes, nodeCount, color))
            return VK_TRUE;
    }
    color[node] = 2; /* BLACK - done */
    return VK_FALSE;
}

VkBool32 vk_ml_validate_graph_create(
    const VkMLGraphCreateInfoKHR *pCreateInfo,
    const VkPhysicalDeviceMLFeaturesKHR *features,
    const VkPhysicalDeviceMLPropertiesKHR *props)
{
    if (!pCreateInfo || !features || !props)
        return VK_FALSE;
    if ((uint32_t)pCreateInfo->sType != VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR)
        return VK_FALSE;

    /* VUID_ML_GRAPH_FEATURE */
    if (!features->mlGraph)
        return VK_FALSE;

    /* VUID_ML_GRAPH_NODE_COUNT */
    if (pCreateInfo->nodeCount == 0 || pCreateInfo->nodeCount > props->maxMLGraphNodeCount)
        return VK_FALSE;

    /* VUID_ML_GRAPH_INPUT_COUNT */
    if (pCreateInfo->externalInputCount == 0)
        return VK_FALSE;

    /* VUID_ML_GRAPH_OUTPUT_COUNT */
    if (pCreateInfo->externalOutputCount == 0)
        return VK_FALSE;

    if (!pCreateInfo->pNodes)
        return VK_FALSE;

    /* VUID_ML_GRAPH_DAG - cycle detection via DFS */
    uint8_t *color = (uint8_t *)malloc(pCreateInfo->nodeCount * sizeof(uint8_t));
    if (!color)
        return VK_FALSE;
    memset(color, 0, pCreateInfo->nodeCount * sizeof(uint8_t));
    for (uint32_t i = 0; i < pCreateInfo->nodeCount; i++) {
        if (color[i] == 0 && dfs_has_cycle(i, pCreateInfo->pNodes,
                                            pCreateInfo->nodeCount, color)) {
            free(color);
            return VK_FALSE;
        }
    }
    free(color);

    /* Per-node primitive descriptor validation */
    VkPhysicalDeviceMLFeaturesKHR feat_copy = *features;
    for (uint32_t i = 0; i < pCreateInfo->nodeCount; i++) {
        const VkMLGraphNodeCreateInfoKHR *node = &pCreateInfo->pNodes[i];
        if (!node->pOperationDesc)
            continue;
        const VkStructureType *pSType = (const VkStructureType *)node->pOperationDesc;
        switch ((uint32_t)*pSType) {
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR:
            if (!vk_ml_validate_convolution_desc(
                    (const VkMLPrimitiveDescConvolutionKHR *)node->pOperationDesc, &feat_copy))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR:
            if (!vk_ml_validate_gemm_desc(
                    (const VkMLPrimitiveDescGemmKHR *)node->pOperationDesc, &feat_copy))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR:
            if (!vk_ml_validate_pooling_desc(
                    (const VkMLPrimitiveDescPoolingKHR *)node->pOperationDesc))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR:
            if (!vk_ml_validate_activation_desc(
                    (const VkMLPrimitiveDescActivationKHR *)node->pOperationDesc))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR:
            if (!vk_ml_validate_normalization_desc(
                    (const VkMLPrimitiveDescNormalizationKHR *)node->pOperationDesc, &feat_copy))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR:
            if (!vk_ml_validate_elementwise_desc(
                    (const VkMLPrimitiveDescElementwiseKHR *)node->pOperationDesc, &feat_copy))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONCAT_KHR:
            if (!vk_ml_validate_concat_desc(
                    (const VkMLPrimitiveDescConcatKHR *)node->pOperationDesc))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESHAPE_KHR:
            if (!vk_ml_validate_reshape_desc(
                    (const VkMLPrimitiveDescReshapeKHR *)node->pOperationDesc))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_TRANSPOSE_KHR:
            if (!vk_ml_validate_transpose_desc(
                    (const VkMLPrimitiveDescTransposeKHR *)node->pOperationDesc))
                return VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESIZE_KHR:
            if (!vk_ml_validate_resize_desc(
                    (const VkMLPrimitiveDescResizeKHR *)node->pOperationDesc))
                return VK_FALSE;
            break;
        default:
            return VK_FALSE;
        }
    }

    return VK_TRUE;
}

VkBool32 vk_ml_validate_convolution_desc(
    const VkMLPrimitiveDescConvolutionKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR)
        return VK_FALSE;

    /* VUID_CONV_KERNEL */
    if (desc->kernelWidth == 0 || desc->kernelHeight == 0)
        return VK_FALSE;

    /* VUID_CONV_STRIDE */
    if (desc->strideX == 0 || desc->strideY == 0)
        return VK_FALSE;

    /* VUID_CONV_DILATION */
    if (desc->dilationX == 0 || desc->dilationY == 0)
        return VK_FALSE;

    /* VUID_CONV_PADDING - padding fields must be 0 when not EXPLICIT */
    if (desc->paddingMode != VK_ML_PADDING_MODE_EXPLICIT_KHR) {
        if (desc->paddingTop != 0 || desc->paddingBottom != 0 ||
            desc->paddingLeft != 0 || desc->paddingRight != 0)
            return VK_FALSE;
    }

    /* VUID_CONV_FUSED_ACT */
    if (desc->fusedActivation != VK_ML_ACTIVATION_FUNCTION_NONE_KHR &&
        !features->fusedActivations)
        return VK_FALSE;

    /* VUID_CONV_GROUP_COUNT */
    if (desc->groupCount == 0)
        return VK_FALSE;
    /* Divisibility check (inputChannels % groupCount == 0) deferred to dispatch
       time when tensor bindings provide the actual input channel dimension. */

    return VK_TRUE;
}

VkBool32 vk_ml_validate_gemm_desc(
    const VkMLPrimitiveDescGemmKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR)
        return VK_FALSE;

    /* VUID_GEMM_ALPHA */
    if (!isfinite(desc->alpha))
        return VK_FALSE;

    /* VUID_GEMM_BETA */
    if (!isfinite(desc->beta))
        return VK_FALSE;

    /* VUID_GEMM_FUSED_ACT */
    if (desc->fusedActivation != VK_ML_ACTIVATION_FUNCTION_NONE_KHR &&
        !features->fusedActivations)
        return VK_FALSE;

    /* VUID_GEMM_DIMS - requires tensor shape analysis; stub */
    /* VUID_GEMM_BIAS - requires bias tensor check when beta != 0; stub */

    return VK_TRUE;
}

VkBool32 vk_ml_validate_pooling_desc(
    const VkMLPrimitiveDescPoolingKHR *desc)
{
    if (!desc)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR)
        return VK_FALSE;

    /* VUID_POOL_TYPE */
    switch (desc->poolType) {
    case VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR:
    case VK_ML_OPERATION_TYPE_AVERAGE_POOL_2D_KHR:
    case VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR:
        break;
    default:
        return VK_FALSE;
    }

    if (desc->poolType != VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR) {
        /* VUID_POOL_WINDOW */
        if (desc->windowWidth == 0 || desc->windowHeight == 0)
            return VK_FALSE;

        /* VUID_POOL_STRIDE */
        if (desc->strideX == 0 || desc->strideY == 0)
            return VK_FALSE;
    }

    return VK_TRUE;
}

VkBool32 vk_ml_validate_normalization_desc(
    const VkMLPrimitiveDescNormalizationKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR)
        return VK_FALSE;

    /* VUID_NORM_EPSILON */
    if (desc->epsilon <= 0.0f || !isfinite(desc->epsilon))
        return VK_FALSE;

    /* VUID_NORM_TYPE */
    switch (desc->normType) {
    case VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR:
    case VK_ML_OPERATION_TYPE_LRN_KHR:
        break;
    default:
        return VK_FALSE;
    }

    /* VUID_NORM_FUSED_ACT */
    if (desc->fusedActivation != VK_ML_ACTIVATION_FUNCTION_NONE_KHR &&
        !features->fusedActivations)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_elementwise_desc(
    const VkMLPrimitiveDescElementwiseKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR)
        return VK_FALSE;

    /* VUID_ELEM_OP_TYPE */
    switch (desc->opType) {
    case VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR:
    case VK_ML_OPERATION_TYPE_ELEMENTWISE_MUL_KHR:
        break;
    default:
        return VK_FALSE;
    }

    /* VUID_ELEM_FUSED_ACT */
    if (desc->fusedActivation != VK_ML_ACTIVATION_FUNCTION_NONE_KHR &&
        !features->fusedActivations)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_activation_desc(
    const VkMLPrimitiveDescActivationKHR *desc)
{
    if (!desc)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR)
        return VK_FALSE;

    if ((uint32_t)desc->activationType > VK_ML_ACTIVATION_FUNCTION_CLAMP_KHR)
        return VK_FALSE;

    if (desc->activationType == VK_ML_ACTIVATION_FUNCTION_CLAMP_KHR &&
        desc->param0 > desc->param1)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_concat_desc(
    const VkMLPrimitiveDescConcatKHR *desc)
{
    if (!desc)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONCAT_KHR)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_reshape_desc(
    const VkMLPrimitiveDescReshapeKHR *desc)
{
    if (!desc)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESHAPE_KHR)
        return VK_FALSE;

    if (desc->dimensionCount > 0 && !desc->pOutputDimensions)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_transpose_desc(
    const VkMLPrimitiveDescTransposeKHR *desc)
{
    if (!desc)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_TRANSPOSE_KHR)
        return VK_FALSE;

    if (desc->dimensionCount > 0 && !desc->pPermutation)
        return VK_FALSE;

    return VK_TRUE;
}

VkBool32 vk_ml_validate_resize_desc(
    const VkMLPrimitiveDescResizeKHR *desc)
{
    if (!desc)
        return VK_FALSE;
    if ((uint32_t)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESIZE_KHR)
        return VK_FALSE;

    if (desc->scaleHeight <= 0.0f || desc->scaleWidth <= 0.0f)
        return VK_FALSE;

    return VK_TRUE;
}
