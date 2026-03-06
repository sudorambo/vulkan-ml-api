/**
 * @file graph_validation.c
 * @brief ML graph and primitive descriptor validation for VK_KHR_ml_primitives.
 */

#include "vk_ml_validation.h"
#include "../../src/internal.h"

#include <math.h>
#include <stddef.h>
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
    if ((int)pCreateInfo->sType != VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR)
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
    if (pCreateInfo->pNodes) {
        uint8_t color[VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT];
        memset(color, 0, sizeof(color));
        for (uint32_t i = 0; i < pCreateInfo->nodeCount; i++) {
            if (color[i] == 0 && dfs_has_cycle(i, pCreateInfo->pNodes,
                                                pCreateInfo->nodeCount, color))
                return VK_FALSE;
        }
    }
    /* VUID_ML_GRAPH_EDGE_COMPAT - stub: edge compatibility requires shape analysis */

    return VK_TRUE;
}

VkBool32 vk_ml_validate_convolution_desc(
    const VkMLPrimitiveDescConvolutionKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
        return VK_FALSE;
    if ((int)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR)
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
    /* TODO: divisibility check requires input channel dimension from tensor binding */

    return VK_TRUE;
}

VkBool32 vk_ml_validate_gemm_desc(
    const VkMLPrimitiveDescGemmKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
        return VK_FALSE;
    if ((int)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR)
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
    if ((int)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR)
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
    if ((int)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR)
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
    if ((int)desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR)
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
