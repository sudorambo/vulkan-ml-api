/**
 * @file graph_validation.c
 * @brief ML graph and primitive descriptor validation for VK_KHR_ml_primitives.
 */

#include "../validation/vk_ml_validation.h"
#include "../../src/internal.h"

#include <math.h>
#include <stddef.h>

VkBool32 vk_ml_validate_graph_create(
    const VkMLGraphCreateInfoKHR *pCreateInfo,
    const VkPhysicalDeviceMLFeaturesKHR *features,
    const VkPhysicalDeviceMLPropertiesKHR *props)
{
    if (!pCreateInfo || !features || !props)
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

    /* VUID_ML_GRAPH_DAG - stub: full DAG validation requires graph traversal */
    /* VUID_ML_GRAPH_EDGE_COMPAT - stub: edge compatibility requires shape analysis */

    return VK_TRUE;
}

VkBool32 vk_ml_validate_convolution_desc(
    const VkMLPrimitiveDescConvolutionKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
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

    /* VUID_CONV_GROUP_COUNT - requires input channel dimension from tensor binding; stub */

    return VK_TRUE;
}

VkBool32 vk_ml_validate_gemm_desc(
    const VkMLPrimitiveDescGemmKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
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

    /* VUID_POOL_WINDOW */
    if (desc->windowWidth == 0 || desc->windowHeight == 0)
        return VK_FALSE;

    /* VUID_POOL_STRIDE */
    if (desc->strideX == 0 || desc->strideY == 0)
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

    return VK_TRUE;
}

VkBool32 vk_ml_validate_normalization_desc(
    const VkMLPrimitiveDescNormalizationKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!desc || !features)
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
