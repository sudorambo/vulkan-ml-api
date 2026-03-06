/**
 * @file ml_primitives.c
 * @brief VK_KHR_ml_primitives ML primitive descriptor validation.
 */

#include "internal.h"
#include <math.h>

/* ------------------------------------------------------------------ */
/* Primitive descriptor validation                                    */
/* ------------------------------------------------------------------ */

static int is_finite_float(float f)
{
    return isfinite(f);
}

VkBool32 vk_ml_validate_primitive_desc(VkMLOperationTypeKHR opType, const void *pDesc)
{
    if (!pDesc)
        return VK_FALSE;

    switch (opType) {
    case VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR:
    case VK_ML_OPERATION_TYPE_DECONVOLUTION_2D_KHR: {
        const VkMLPrimitiveDescConvolutionKHR *d = (const VkMLPrimitiveDescConvolutionKHR *)pDesc;
        if (d->strideX == 0 || d->strideY == 0)
            return VK_FALSE;
        if (d->dilationX == 0 || d->dilationY == 0)
            return VK_FALSE;
        if (!is_finite_float(d->activationParam0) || !is_finite_float(d->activationParam1))
            return VK_FALSE;
        if (d->groupCount == 0)
            return VK_FALSE;
        break;
    }
    case VK_ML_OPERATION_TYPE_GEMM_KHR:
    case VK_ML_OPERATION_TYPE_FULLY_CONNECTED_KHR: {
        const VkMLPrimitiveDescGemmKHR *d = (const VkMLPrimitiveDescGemmKHR *)pDesc;
        if (!is_finite_float(d->alpha) || !is_finite_float(d->beta))
            return VK_FALSE;
        break;
    }
    case VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR:
    case VK_ML_OPERATION_TYPE_AVERAGE_POOL_2D_KHR:
    case VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR: {
        const VkMLPrimitiveDescPoolingKHR *d = (const VkMLPrimitiveDescPoolingKHR *)pDesc;
        if (d->poolType == VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR ||
            d->poolType == VK_ML_OPERATION_TYPE_AVERAGE_POOL_2D_KHR) {
            if (d->windowWidth == 0 || d->windowHeight == 0)
                return VK_FALSE;
            if (d->strideX == 0 || d->strideY == 0)
                return VK_FALSE;
        }
        break;
    }
    case VK_ML_OPERATION_TYPE_RELU_KHR:
    case VK_ML_OPERATION_TYPE_SIGMOID_KHR:
    case VK_ML_OPERATION_TYPE_TANH_KHR:
    case VK_ML_OPERATION_TYPE_LEAKY_RELU_KHR:
    case VK_ML_OPERATION_TYPE_PRELU_KHR:
    case VK_ML_OPERATION_TYPE_SOFTMAX_KHR: {
        const VkMLPrimitiveDescActivationKHR *d = (const VkMLPrimitiveDescActivationKHR *)pDesc;
        if (!is_finite_float(d->param0) || !is_finite_float(d->param1))
            return VK_FALSE;
        break;
    }
    case VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR:
    case VK_ML_OPERATION_TYPE_LRN_KHR: {
        const VkMLPrimitiveDescNormalizationKHR *d =
            (const VkMLPrimitiveDescNormalizationKHR *)pDesc;
        if (d->epsilon <= 0.0f || !is_finite_float(d->epsilon))
            return VK_FALSE;
        if (!is_finite_float(d->activationParam0) || !is_finite_float(d->activationParam1))
            return VK_FALSE;
        break;
    }
    case VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR:
    case VK_ML_OPERATION_TYPE_ELEMENTWISE_MUL_KHR: {
        const VkMLPrimitiveDescElementwiseKHR *d = (const VkMLPrimitiveDescElementwiseKHR *)pDesc;
        if (!is_finite_float(d->activationParam0) || !is_finite_float(d->activationParam1))
            return VK_FALSE;
        break;
    }
    case VK_ML_OPERATION_TYPE_CONCAT_KHR: {
        const VkMLPrimitiveDescConcatKHR *d = (const VkMLPrimitiveDescConcatKHR *)pDesc;
        if ((int)d->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONCAT_KHR)
            return VK_FALSE;
        break;
    }
    case VK_ML_OPERATION_TYPE_RESHAPE_KHR: {
        const VkMLPrimitiveDescReshapeKHR *d = (const VkMLPrimitiveDescReshapeKHR *)pDesc;
        if ((int)d->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESHAPE_KHR)
            return VK_FALSE;
        if (d->dimensionCount == 0 || !d->pOutputDimensions)
            return VK_FALSE;
        for (uint32_t i = 0; i < d->dimensionCount; i++) {
            if (d->pOutputDimensions[i] == 0)
                return VK_FALSE;
        }
        break;
    }
    case VK_ML_OPERATION_TYPE_TRANSPOSE_KHR: {
        const VkMLPrimitiveDescTransposeKHR *d = (const VkMLPrimitiveDescTransposeKHR *)pDesc;
        if ((int)d->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_TRANSPOSE_KHR)
            return VK_FALSE;
        if (d->dimensionCount == 0 || !d->pPermutation)
            return VK_FALSE;
        break;
    }
    case VK_ML_OPERATION_TYPE_RESIZE_KHR: {
        const VkMLPrimitiveDescResizeKHR *d = (const VkMLPrimitiveDescResizeKHR *)pDesc;
        if ((int)d->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESIZE_KHR)
            return VK_FALSE;
        if (d->scaleHeight <= 0.0f || d->scaleWidth <= 0.0f)
            return VK_FALSE;
        if (!is_finite_float(d->scaleHeight) || !is_finite_float(d->scaleWidth))
            return VK_FALSE;
        break;
    }
    default:
        return VK_FALSE;
    }
    return VK_TRUE;
}
