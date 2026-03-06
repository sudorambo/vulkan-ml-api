/**
 * @file vk_ml_validation.h
 * @brief Validation layer declarations for VK_KHR_ml_primitives.
 *
 * All validation functions return VkBool32: VK_TRUE = valid, VK_FALSE = invalid.
 */

#ifndef VK_ML_VALIDATION_H_
#define VK_ML_VALIDATION_H_

#include <vulkan/vulkan_ml_primitives.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct VkTensorKHR_T;

/* Tensor validation */
VkBool32 vk_ml_validate_tensor_create(
    const VkTensorCreateInfoKHR *pCreateInfo,
    const VkPhysicalDeviceMLFeaturesKHR *features,
    const VkPhysicalDeviceMLPropertiesKHR *props);

VkBool32 vk_ml_validate_tensor_view_create(
    const VkTensorViewCreateInfoKHR *pCreateInfo,
    const struct VkTensorKHR_T *tensor);

VkBool32 vk_ml_validate_tensor_bind(
    const VkBindTensorMemoryInfoKHR *pBindInfo,
    const VkPhysicalDeviceMLPropertiesKHR *props);

VkBool32 vk_ml_validate_tensor_copy(
    const VkCopyTensorInfoKHR *pCopyInfo);

/* Graph validation */
VkBool32 vk_ml_validate_graph_create(
    const VkMLGraphCreateInfoKHR *pCreateInfo,
    const VkPhysicalDeviceMLFeaturesKHR *features,
    const VkPhysicalDeviceMLPropertiesKHR *props);

VkBool32 vk_ml_validate_convolution_desc(
    const VkMLPrimitiveDescConvolutionKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features);

VkBool32 vk_ml_validate_gemm_desc(
    const VkMLPrimitiveDescGemmKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features);

VkBool32 vk_ml_validate_pooling_desc(
    const VkMLPrimitiveDescPoolingKHR *desc);

VkBool32 vk_ml_validate_normalization_desc(
    const VkMLPrimitiveDescNormalizationKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features);

VkBool32 vk_ml_validate_elementwise_desc(
    const VkMLPrimitiveDescElementwiseKHR *desc,
    const VkPhysicalDeviceMLFeaturesKHR *features);

/* Session validation */
VkBool32 vk_ml_validate_session_create(
    const VkMLSessionCreateInfoKHR *pCreateInfo,
    VkDeviceSize requiredScratchSize,
    const VkPhysicalDeviceMLFeaturesKHR *features);

/* Dispatch validation */
VkBool32 vk_ml_validate_dispatch(
    const VkMLGraphDispatchInfoKHR *pDispatchInfo);

#ifdef __cplusplus
}
#endif

#endif /* VK_ML_VALIDATION_H_ */
