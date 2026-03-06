/**
 * @file feature_query.c
 * @brief VK_KHR_ml_primitives feature and property query implementation.
 *
 * Implements VkPhysicalDeviceMLFeaturesKHR and VkPhysicalDeviceMLPropertiesKHR
 * population, plus tensor format support queries for the reference implementation.
 */

#include "internal.h"

/* ------------------------------------------------------------------ */
/* Feature population                                                  */
/* ------------------------------------------------------------------ */

void vk_ml_populate_features(VkPhysicalDeviceMLFeaturesKHR *features)
{
    if (!features)
        return;

    features->mlPrimitives = VK_TRUE;
    features->mlGraph = VK_TRUE;
    features->tensorObjects = VK_TRUE;
    features->tensorShaderAccess = VK_TRUE;
    features->tensorImageAliasing = VK_TRUE;
    features->fp16Tensors = VK_TRUE;
    features->bf16Tensors = VK_TRUE;
    features->int8Tensors = VK_TRUE;
    features->int4Tensors = VK_TRUE;
    features->fp8Tensors = VK_TRUE;
    features->fusedActivations = VK_TRUE;
    features->mlGraphScratchAutoAllocation = VK_TRUE;
}

/* ------------------------------------------------------------------ */
/* Property population                                                 */
/* ------------------------------------------------------------------ */

void vk_ml_populate_properties(VkPhysicalDeviceMLPropertiesKHR *props)
{
    if (!props)
        return;

    props->maxTensorDimensions = VK_ML_REF_MAX_TENSOR_DIMENSIONS;
    props->maxTensorElements = VK_ML_REF_MAX_TENSOR_ELEMENTS;
    props->maxTensorDimensionSize = VK_ML_REF_MAX_TENSOR_DIMENSION_SIZE;
    props->maxMLGraphNodeCount = VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT;
    props->maxMLGraphDepth = VK_ML_REF_MAX_ML_GRAPH_DEPTH;
    props->maxMLSessionCount = VK_ML_REF_MAX_ML_SESSION_COUNT;
    props->maxConcurrentMLDispatches = VK_ML_REF_MAX_CONCURRENT_DISPATCHES;
    props->supportedPrimitiveCount = VK_ML_REF_SUPPORTED_PRIMITIVE_COUNT;
    props->minTensorMemoryAlignment = VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN;
    props->maxScratchMemorySize = VK_ML_REF_MAX_SCRATCH_MEMORY_SIZE;
}

/* ------------------------------------------------------------------ */
/* Tensor format support                                               */
/* ------------------------------------------------------------------ */

static const VkFormat vk_ml_supported_tensor_formats[] = {
    VK_FORMAT_R8_SINT,
    VK_FORMAT_R8_UINT,
    VK_FORMAT_R16_SFLOAT,
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R16_SINT,
    VK_FORMAT_R16_UINT,
    VK_FORMAT_R32_SINT,
    VK_FORMAT_R32_UINT,
    (VkFormat)VK_FORMAT_R16_BFLOAT_KHR,
    (VkFormat)VK_FORMAT_R8_E4M3_KHR,
    (VkFormat)VK_FORMAT_R8_E5M2_KHR,
    (VkFormat)VK_FORMAT_R8_BOOL,
};

#define VK_ML_SUPPORTED_TENSOR_FORMAT_COUNT \
    (sizeof(vk_ml_supported_tensor_formats) / sizeof(vk_ml_supported_tensor_formats[0]))

VkBool32 vk_ml_is_tensor_format_supported(VkFormat format)
{
    for (size_t i = 0; i < VK_ML_SUPPORTED_TENSOR_FORMAT_COUNT; i++) {
        if (vk_ml_supported_tensor_formats[i] == format)
            return VK_TRUE;
    }
    return VK_FALSE;
}

void vk_ml_populate_tensor_format_properties(VkFormat format,
                                             VkTensorFormatPropertiesKHR *props)
{
    if (!props)
        return;

    if (vk_ml_is_tensor_format_supported(format)) {
        props->tensorFeatures =
            (VkFormatFeatureFlags2)VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_KHR |
            (VkFormatFeatureFlags2)VK_FORMAT_FEATURE_2_TENSOR_IMAGE_ALIASING_BIT_KHR;
    } else {
        props->tensorFeatures = 0;
    }
}
