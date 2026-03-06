/**
 * @file internal.h
 * @brief Internal shared declarations for the VK_KHR_ml_primitives
 *        reference implementation.
 *
 * NOT part of the public API. Implementation-private structures and
 * helpers only.
 */

#ifndef VK_ML_INTERNAL_H_
#define VK_ML_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan_ml_primitives.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Internal object representations                                     */
/* ------------------------------------------------------------------ */

typedef struct VkTensorKHR_T {
    VkTensorDescriptionKHR description;
    uint32_t *dimensions;
    VkDeviceSize *strides;
    VkSharingMode sharingMode;
    uint32_t queueFamilyIndexCount;
    uint32_t *queueFamilyIndices;
    VkDeviceMemory boundMemory;
    VkDeviceSize memoryOffset;
    VkBool32 memoryBound;
} VkTensorKHR_T;

typedef struct VkTensorViewKHR_T {
    VkTensorKHR tensor;
    VkFormat format;
    uint32_t dimensionCount;
    uint32_t *dimensionOffsets;
    uint32_t *dimensionSizes;
} VkTensorViewKHR_T;

typedef struct VkMLGraphKHR_T {
    uint32_t nodeCount;
    VkMLGraphNodeCreateInfoKHR *nodes;
    uint32_t externalInputCount;
    VkTensorDescriptionKHR *externalInputDescs;
    uint32_t externalOutputCount;
    VkTensorDescriptionKHR *externalOutputDescs;
    uint32_t constantWeightCount;
    VkTensorDescriptionKHR *constantWeightDescs;
    VkDeviceSize scratchMemorySize;
} VkMLGraphKHR_T;

typedef struct VkMLSessionKHR_T {
    VkMLGraphKHR graph;
    VkDeviceMemory scratchMemory;
    VkDeviceSize scratchMemoryOffset;
    VkDeviceSize scratchMemorySize;
    VkBool32 autoAllocated;
    void *autoScratchHost;
} VkMLSessionKHR_T;

/* ------------------------------------------------------------------ */
/* Reference implementation limits                                     */
/* ------------------------------------------------------------------ */

#define VK_ML_REF_MAX_TENSOR_DIMENSIONS     8
#define VK_ML_REF_MAX_TENSOR_ELEMENTS       (1ULL << 32)
#define VK_ML_REF_MAX_TENSOR_DIMENSION_SIZE (1U << 16)
#define VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT   256
#define VK_ML_REF_MAX_ML_GRAPH_DEPTH        128
#define VK_ML_REF_MAX_ML_SESSION_COUNT      64
#define VK_ML_REF_MAX_CONCURRENT_DISPATCHES 16
#define VK_ML_REF_SUPPORTED_PRIMITIVE_COUNT 21
#define VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN   64
#define VK_ML_REF_MAX_SCRATCH_MEMORY_SIZE   (1ULL << 30)

/* Portable max alignment (MSVC C mode lacks _Alignof/max_align_t) */
#if defined(_MSC_VER)
#define VK_ML_MAX_ALIGNMENT 16
#else
#define VK_ML_MAX_ALIGNMENT _Alignof(max_align_t)
#endif

/* ------------------------------------------------------------------ */
/* Allocation helpers                                                  */
/* ------------------------------------------------------------------ */

static inline void *vk_ml_alloc(const VkAllocationCallbacks *pAllocator, size_t size)
{
    if (size == 0)
        return NULL;
    if (pAllocator && pAllocator->pfnAllocation) {
        return pAllocator->pfnAllocation(pAllocator->pUserData, size, VK_ML_MAX_ALIGNMENT,
                                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    return malloc(size);
}

static inline void vk_ml_free(const VkAllocationCallbacks *pAllocator, void *ptr)
{
    if (!ptr)
        return;
    if (pAllocator && pAllocator->pfnFree) {
        pAllocator->pfnFree(pAllocator->pUserData, ptr);
        return;
    }
    free(ptr);
}

/* ------------------------------------------------------------------ */
/* Feature query helpers                                               */
/* ------------------------------------------------------------------ */

void vk_ml_populate_features(VkPhysicalDeviceMLFeaturesKHR *features);
void vk_ml_populate_properties(VkPhysicalDeviceMLPropertiesKHR *props);
VkBool32 vk_ml_is_tensor_format_supported(VkFormat format);
void vk_ml_populate_tensor_format_properties(VkFormat format, VkTensorFormatPropertiesKHR *props);

/* ------------------------------------------------------------------ */
/* Format helpers                                                      */
/* ------------------------------------------------------------------ */

VkBool32 vk_ml_validate_primitive_desc(VkMLOperationTypeKHR opType, const void *pDesc);

static inline uint32_t vk_ml_format_element_size(VkFormat format)
{
    uint32_t f = (uint32_t)format;
    switch (f) {
    case VK_FORMAT_R16_SFLOAT:
        return 2;
    case VK_FORMAT_R32_SFLOAT:
        return 4;
    case VK_FORMAT_R8_SINT:
        return 1;
    case VK_FORMAT_R8_UINT:
        return 1;
    case VK_FORMAT_R16_SINT:
        return 2;
    case VK_FORMAT_R16_UINT:
        return 2;
    case VK_FORMAT_R32_SINT:
        return 4;
    case VK_FORMAT_R32_UINT:
        return 4;
    case (uint32_t)VK_FORMAT_R16_BFLOAT_KHR:
        return 2;
    case (uint32_t)VK_FORMAT_R8_E4M3_KHR:
        return 1;
    case (uint32_t)VK_FORMAT_R8_E5M2_KHR:
        return 1;
    case (uint32_t)VK_FORMAT_R8_BOOL_KHR:
        return 1;
    default:
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* VUID string constants                                               */
/* ------------------------------------------------------------------ */

#define VUID_FEATURES_STYPE                "VUID-VkPhysicalDeviceMLFeaturesKHR-sType-sType"
#define VUID_PROPERTIES_STYPE              "VUID-VkPhysicalDeviceMLPropertiesKHR-sType-sType"
#define VUID_TENSOR_OBJECTS_FEATURE        "VUID-vkCreateTensorKHR-tensorObjects-00001"
#define VUID_TENSOR_DEVICE_QUEUE           "VUID-vkCreateTensorKHR-device-00002"
#define VUID_TENSOR_CREATE_SHARING_MODE    "VUID-VkTensorCreateInfoKHR-sharingMode-00001"
#define VUID_TENSOR_CREATE_SHARING_INDICES "VUID-VkTensorCreateInfoKHR-sharingMode-00002"
#define VUID_TENSOR_CREATE_DESC            "VUID-VkTensorCreateInfoKHR-pDescription-00003"
#define VUID_TENSOR_DESC_DIM_COUNT         "VUID-VkTensorDescriptionKHR-dimensionCount-00001"
#define VUID_TENSOR_DESC_DIM_VALUES        "VUID-VkTensorDescriptionKHR-pDimensions-00002"
#define VUID_TENSOR_DESC_DIM_PRODUCT       "VUID-VkTensorDescriptionKHR-pDimensions-00003"
#define VUID_TENSOR_DESC_STRIDES_OPTIMAL   "VUID-VkTensorDescriptionKHR-pStrides-00004"
#define VUID_TENSOR_DESC_FORMAT            "VUID-VkTensorDescriptionKHR-format-00005"
#define VUID_TENSOR_DESC_STRIDE_ALIGN      "VUID-VkTensorDescriptionKHR-pStrides-00006"
#define VUID_BIND_TENSOR_ALREADY_BOUND     "VUID-VkBindTensorMemoryInfoKHR-tensor-00001"
#define VUID_BIND_TENSOR_ALIGNMENT         "VUID-VkBindTensorMemoryInfoKHR-memoryOffset-00002"
#define VUID_BIND_TENSOR_MEM_TYPE          "VUID-VkBindTensorMemoryInfoKHR-memory-00003"
#define VUID_BIND_TENSOR_MEM_SIZE          "VUID-VkBindTensorMemoryInfoKHR-size-00004"
#define VUID_DESTROY_TENSOR_IN_USE         "VUID-vkDestroyTensorKHR-tensor-00001"
#define VUID_DESTROY_TENSOR_VALID          "VUID-vkDestroyTensorKHR-tensor-00002"
#define VUID_DESTROY_GRAPH_IN_USE          "VUID-vkDestroyMLGraphKHR-graph-00001"
#define VUID_DESTROY_GRAPH_VALID           "VUID-vkDestroyMLGraphKHR-graph-00002"
#define VUID_DESTROY_SESSION_IN_USE        "VUID-vkDestroyMLSessionKHR-session-00001"
#define VUID_COPY_TENSOR_REGION_COUNT      "VUID-VkCopyTensorInfoKHR-regionCount-00001"
#define VUID_COPY_TENSOR_SRC_OFFSETS       "VUID-VkTensorCopyKHR-pSrcOffsets-00001"
#define VUID_COPY_TENSOR_DST_OFFSETS       "VUID-VkTensorCopyKHR-pDstOffsets-00002"
#define VUID_COPY_TENSOR_CMD_STATE         "VUID-vkCmdCopyTensorKHR-commandBuffer-00001"
#define VUID_COPY_TENSOR_SRC_USAGE         "VUID-vkCmdCopyTensorKHR-srcTensor-00002"
#define VUID_COPY_TENSOR_DST_USAGE         "VUID-vkCmdCopyTensorKHR-dstTensor-00003"
#define VUID_COPY_TENSOR_MEM_BOUND         "VUID-vkCmdCopyTensorKHR-srcTensor-00004"
#define VUID_COPY_TENSOR_FORMAT            "VUID-vkCmdCopyTensorKHR-format-00005"
#define VUID_COPY_TENSOR_SAME              "VUID-vkCmdCopyTensorKHR-srcTensor-00006"
#define VUID_TENSOR_VIEW_OBJECTS_FEATURE   "VUID-vkCreateTensorViewKHR-tensorObjects-00001"
#define VUID_TENSOR_VIEW_TENSOR_HANDLE     "VUID-vkCreateTensorViewKHR-tensor-00002"
#define VUID_TENSOR_VIEW_STYPE             "VUID-VkTensorViewCreateInfoKHR-sType-sType"
#define VUID_TENSOR_VIEW_MEMORY_BOUND      "VUID-VkTensorViewCreateInfoKHR-tensor-00004"
#define VUID_TENSOR_VIEW_FORMAT            "VUID-VkTensorViewCreateInfoKHR-format-00001"
#define VUID_TENSOR_VIEW_RANGE             "VUID-VkTensorViewCreateInfoKHR-range-00002"
#define VUID_TENSOR_VIEW_DIM_COUNT         "VUID-VkTensorViewCreateInfoKHR-dimensionCount-00003"
#define VUID_TENSOR_VIEW_HANDLE            "VUID-VkTensorViewCreateInfoKHR-tensor-00004"
#define VUID_ML_GRAPH_FEATURE              "VUID-VkMLGraphCreateInfoKHR-mlGraph-00001"
#define VUID_ML_GRAPH_NODE_COUNT           "VUID-VkMLGraphCreateInfoKHR-nodeCount-00002"
#define VUID_ML_GRAPH_DAG                  "VUID-VkMLGraphCreateInfoKHR-pNodes-00003"
#define VUID_ML_GRAPH_EDGE_COMPAT          "VUID-VkMLGraphCreateInfoKHR-pNodes-00004"
#define VUID_ML_GRAPH_INPUT_COUNT          "VUID-VkMLGraphCreateInfoKHR-externalInputCount-00005"
#define VUID_ML_GRAPH_OUTPUT_COUNT         "VUID-VkMLGraphCreateInfoKHR-externalOutputCount-00006"
#define VUID_CONV_STRIDE                   "VUID-VkMLPrimitiveDescConvolutionKHR-strideX-00001"
#define VUID_CONV_DILATION                 "VUID-VkMLPrimitiveDescConvolutionKHR-dilationX-00002"
#define VUID_CONV_PADDING                  "VUID-VkMLPrimitiveDescConvolutionKHR-paddingMode-00003"
#define VUID_CONV_FUSED_ACT                "VUID-VkMLPrimitiveDescConvolutionKHR-fusedActivation-00004"
#define VUID_CONV_GROUP_COUNT              "VUID-VkMLPrimitiveDescConvolutionKHR-groupCount-00005"
#define VUID_GEMM_ALPHA                    "VUID-VkMLPrimitiveDescGemmKHR-alpha-00001"
#define VUID_GEMM_BETA                     "VUID-VkMLPrimitiveDescGemmKHR-beta-00002"
#define VUID_GEMM_DIMS                     "VUID-VkMLPrimitiveDescGemmKHR-transposeA-00003"
#define VUID_GEMM_FUSED_ACT                "VUID-VkMLPrimitiveDescGemmKHR-fusedActivation-00004"
#define VUID_GEMM_BIAS                     "VUID-VkMLPrimitiveDescGemmKHR-beta-00005"
#define VUID_POOL_WINDOW                   "VUID-VkMLPrimitiveDescPoolingKHR-windowWidth-00001"
#define VUID_POOL_STRIDE                   "VUID-VkMLPrimitiveDescPoolingKHR-strideX-00002"
#define VUID_POOL_TYPE                     "VUID-VkMLPrimitiveDescPoolingKHR-poolType-00003"
#define VUID_NORM_EPSILON                  "VUID-VkMLPrimitiveDescNormalizationKHR-epsilon-00001"
#define VUID_NORM_TYPE                     "VUID-VkMLPrimitiveDescNormalizationKHR-normType-00002"
#define VUID_NORM_FUSED_ACT                "VUID-VkMLPrimitiveDescNormalizationKHR-fusedActivation-00003"
#define VUID_ELEM_OP_TYPE                  "VUID-VkMLPrimitiveDescElementwiseKHR-opType-00001"
#define VUID_ELEM_FUSED_ACT                "VUID-VkMLPrimitiveDescElementwiseKHR-fusedActivation-00002"
#define VUID_SESSION_SCRATCH_SIZE          "VUID-VkMLSessionCreateInfoKHR-scratchMemory-00001"
#define VUID_SESSION_SCRATCH_AUTO          "VUID-VkMLSessionCreateInfoKHR-scratchMemory-00002"
#define VUID_SESSION_GRAPH_VALID           "VUID-VkMLSessionCreateInfoKHR-graph-00003"
#define VUID_SESSION_SCRATCH_OFFSET_ALIGN  "VUID-VkMLSessionCreateInfoKHR-scratchMemoryOffset-00004"
#define VUID_DISPATCH_CMD_STATE            "VUID-vkCmdDispatchMLGraphKHR-commandBuffer-00001"
#define VUID_DISPATCH_COMPUTE_QUEUE        "VUID-vkCmdDispatchMLGraphKHR-commandBuffer-00002"
#define VUID_DISPATCH_SESSION              "VUID-vkCmdDispatchMLGraphKHR-session-00003"
#define VUID_DISPATCH_INPUT_COUNT          "VUID-vkCmdDispatchMLGraphKHR-inputTensorCount-00004"
#define VUID_DISPATCH_OUTPUT_COUNT         "VUID-vkCmdDispatchMLGraphKHR-outputTensorCount-00005"
#define VUID_DISPATCH_WEIGHT_COUNT         "VUID-vkCmdDispatchMLGraphKHR-weightTensorCount-00006"
#define VUID_DISPATCH_INPUT_USAGE          "VUID-vkCmdDispatchMLGraphKHR-pInputTensors-00007"
#define VUID_DISPATCH_OUTPUT_USAGE         "VUID-vkCmdDispatchMLGraphKHR-pOutputTensors-00008"
#define VUID_DISPATCH_WEIGHT_USAGE         "VUID-vkCmdDispatchMLGraphKHR-pWeightTensors-00009"
#define VUID_DISPATCH_INPUT_MEM_BOUND      "VUID-vkCmdDispatchMLGraphKHR-pInputTensors-00010"
#define VUID_DISPATCH_OUTPUT_MEM_BOUND     "VUID-vkCmdDispatchMLGraphKHR-pOutputTensors-00011"
#define VUID_DISPATCH_WEIGHT_MEM_BOUND     "VUID-vkCmdDispatchMLGraphKHR-pWeightTensors-00012"
#define VUID_DISPATCH_INPUT_SHAPES         "VUID-vkCmdDispatchMLGraphKHR-shapes-00013"
#define VUID_DISPATCH_OUTPUT_SHAPES        "VUID-vkCmdDispatchMLGraphKHR-shapes-00014"
#define VUID_DESCRIPTOR_TENSOR_COUNT       "VUID-VkWriteDescriptorSetTensorKHR-tensorCount-00001"
#define VUID_DESCRIPTOR_TENSOR_VIEWS       "VUID-VkWriteDescriptorSetTensorKHR-pTensorViews-00002"

#ifdef __cplusplus
}
#endif

#endif /* VK_ML_INTERNAL_H_ */
