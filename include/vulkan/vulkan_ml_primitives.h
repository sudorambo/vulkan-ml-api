/**
 * @file vulkan_ml_primitives.h
 * @brief VK_KHR_ml_primitives Vulkan extension public header.
 *
 * Defines tensor resource types, ML primitive operations, ML graph
 * compilation, and ML session/dispatch APIs for GPU-accelerated
 * machine learning workloads.
 *
 * Specification: spec/VK_KHR_ml_primitives.adoc
 * Extension revision: 1
 * Required: Vulkan 1.3
 */

#ifndef VULKAN_ML_PRIMITIVES_H_
#define VULKAN_ML_PRIMITIVES_H_

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Extension constants                                                 */
/* ------------------------------------------------------------------ */

#define VK_KHR_ML_PRIMITIVES_SPEC_VERSION   1
#define VK_KHR_ML_PRIMITIVES_EXTENSION_NAME "VK_KHR_ml_primitives"

/* ------------------------------------------------------------------ */
/* Handle types (T006)                                                 */
/* ------------------------------------------------------------------ */

/** @brief Non-dispatchable handle for an N-dimensional tensor resource. */
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkTensorKHR)
/** @brief Non-dispatchable handle for a typed view into a VkTensorKHR. */
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkTensorViewKHR)
/** @brief Non-dispatchable handle for a compiled DAG of ML primitive operations. */
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkMLGraphKHR)
/** @brief Non-dispatchable handle for an execution context for ML graph dispatches. */
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkMLSessionKHR)

/* ------------------------------------------------------------------ */
/* VkStructureType extension values (T004)                             */
/* ------------------------------------------------------------------ */

/* Values are placeholders pending Khronos registry assignment. */
enum {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR       = 1000559000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR     = 1000559001,
    VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR                = 1000559002,
    VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR                = 1000559003,
    VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR           = 1000559004,
    VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR             = 1000559005,
    VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR   = 1000559006,
    VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR           = 1000559007,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_KHR       = 1000559008,
    VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_KHR          = 1000559009,
    VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR                  = 1000559010,
    VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR              = 1000559011,
    VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR         = 1000559012,
    VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR            = 1000559013,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR     = 1000559014,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR            = 1000559015,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR         = 1000559016,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR      = 1000559017,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR   = 1000559018,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR     = 1000559019,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONCAT_KHR          = 1000559020,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESHAPE_KHR         = 1000559021,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_TRANSPOSE_KHR       = 1000559022,
    VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_RESIZE_KHR          = 1000559023,
    VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR            = 1000559024,
    VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR                 = 1000559025,
    VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR            = 1000559026,
    VK_STRUCTURE_TYPE_TENSOR_COPY_KHR                       = 1000559027,
};

/* VkObjectType extensions (T004) */
enum {
    VK_OBJECT_TYPE_TENSOR_KHR      = 1000559000,
    VK_OBJECT_TYPE_TENSOR_VIEW_KHR = 1000559001,
    VK_OBJECT_TYPE_ML_GRAPH_KHR    = 1000559002,
    VK_OBJECT_TYPE_ML_SESSION_KHR  = 1000559003,
};

/* VkDescriptorType extension */
enum {
    VK_DESCRIPTOR_TYPE_TENSOR_KHR = 1000559000,
};

/* VkPipelineStageFlagBits2 extension */
static const VkPipelineStageFlags2 VK_PIPELINE_STAGE_2_ML_GRAPH_BIT_KHR =
    0x100000000ULL;

/* VkAccessFlagBits2 extensions */
static const VkAccessFlags2 VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR  =
    0x100000000ULL;
static const VkAccessFlags2 VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR =
    0x200000000ULL;

/* VkFormatFeatureFlagBits2 extensions */
static const VkFormatFeatureFlags2 VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_KHR =
    0x100000000ULL;
static const VkFormatFeatureFlags2 VK_FORMAT_FEATURE_2_TENSOR_IMAGE_ALIASING_BIT_KHR =
    0x200000000ULL;

/* VkQueueFlagBits extension */
enum {
    VK_QUEUE_ML_BIT_KHR = 0x00000040,
};

/* VkFormat extensions for ML tensor element types */
enum {
    VK_FORMAT_R8_BOOL_KHR    = 1000559000,
    VK_FORMAT_R16_BFLOAT_KHR = 1000559001,
    VK_FORMAT_R8_E4M3_KHR   = 1000559002,
    VK_FORMAT_R8_E5M2_KHR   = 1000559003,
};

/* ------------------------------------------------------------------ */
/* Enumerations (T005)                                                 */
/* ------------------------------------------------------------------ */

/** @brief Data layout strategy for tensor elements in memory. */
typedef enum VkTensorTilingKHR {
    VK_TENSOR_TILING_OPTIMAL_KHR = 0,  /**< Implementation-selected opaque layout; applications must not assume element arrangement. */
    VK_TENSOR_TILING_LINEAR_KHR  = 1,  /**< Row-major order with application-specified or densely-packed strides; memory may be mapped. */
    VK_TENSOR_TILING_MAX_ENUM_KHR = 0x7FFFFFFF
} VkTensorTilingKHR;

/** @brief Bitmask of intended usage for a tensor. */
typedef enum VkTensorUsageFlagBitsKHR {
    VK_TENSOR_USAGE_SHADER_BIT_KHR          = 0x00000001,  /**< May be accessed from compute shaders via SPIR-V tensor accessors. */
    VK_TENSOR_USAGE_TRANSFER_SRC_BIT_KHR    = 0x00000002,  /**< May be used as source in vkCmdCopyTensorKHR. */
    VK_TENSOR_USAGE_TRANSFER_DST_BIT_KHR    = 0x00000004,  /**< May be used as destination in vkCmdCopyTensorKHR. */
    VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR  = 0x00000008,  /**< May be bound as input to an ML graph dispatch. */
    VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR = 0x00000010,  /**< May be bound as output of an ML graph dispatch. */
    VK_TENSOR_USAGE_ML_GRAPH_WEIGHT_BIT_KHR = 0x00000020,  /**< May be bound as constant weight/bias parameter of an ML graph. */
    VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_KHR  = 0x00000040,  /**< May alias device memory with a VkImage; requires tensorImageAliasing feature. */
    VK_TENSOR_USAGE_FLAG_BITS_MAX_ENUM_KHR  = 0x7FFFFFFF
} VkTensorUsageFlagBitsKHR;
typedef VkFlags VkTensorUsageFlagsKHR;

/** @brief Type of ML primitive operation performed by a graph node. */
typedef enum VkMLOperationTypeKHR {
    VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR      = 0,   /**< 2D convolution. */
    VK_ML_OPERATION_TYPE_DECONVOLUTION_2D_KHR    = 1,   /**< 2D transposed convolution. */
    VK_ML_OPERATION_TYPE_GEMM_KHR                = 2,   /**< General matrix multiply. */
    VK_ML_OPERATION_TYPE_FULLY_CONNECTED_KHR     = 3,   /**< Fully connected (dense) layer. */
    VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR         = 4,   /**< 2D max pooling. */
    VK_ML_OPERATION_TYPE_AVERAGE_POOL_2D_KHR     = 5,   /**< 2D average pooling. */
    VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR = 6,   /**< Global average pooling. */
    VK_ML_OPERATION_TYPE_RELU_KHR                = 7,   /**< ReLU activation. */
    VK_ML_OPERATION_TYPE_SIGMOID_KHR             = 8,   /**< Sigmoid activation. */
    VK_ML_OPERATION_TYPE_TANH_KHR                = 9,   /**< Tanh activation. */
    VK_ML_OPERATION_TYPE_LEAKY_RELU_KHR          = 10,  /**< Leaky ReLU activation. */
    VK_ML_OPERATION_TYPE_PRELU_KHR               = 11,  /**< Parametric ReLU activation. */
    VK_ML_OPERATION_TYPE_SOFTMAX_KHR             = 12,  /**< Softmax activation. */
    VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR = 13,  /**< Batch normalization. */
    VK_ML_OPERATION_TYPE_LRN_KHR                 = 14,  /**< Local response normalization. */
    VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR     = 15,  /**< Element-wise addition. */
    VK_ML_OPERATION_TYPE_ELEMENTWISE_MUL_KHR     = 16,  /**< Element-wise multiplication. */
    VK_ML_OPERATION_TYPE_CONCAT_KHR              = 17,  /**< Concatenation along a dimension. */
    VK_ML_OPERATION_TYPE_RESHAPE_KHR             = 18,  /**< Reshape tensor dimensions. */
    VK_ML_OPERATION_TYPE_TRANSPOSE_KHR           = 19,  /**< Transpose dimensions. */
    VK_ML_OPERATION_TYPE_RESIZE_KHR              = 20,  /**< Spatial resize (e.g., bilinear). */
    VK_ML_OPERATION_TYPE_MAX_ENUM_KHR            = 0x7FFFFFFF
} VkMLOperationTypeKHR;

/** @brief Activation function for ML primitives (fused or standalone). */
typedef enum VkMLActivationFunctionKHR {
    VK_ML_ACTIVATION_FUNCTION_NONE_KHR       = 0,  /**< No activation. */
    VK_ML_ACTIVATION_FUNCTION_RELU_KHR       = 1,  /**< ReLU: f(x) = max(0, x). */
    VK_ML_ACTIVATION_FUNCTION_SIGMOID_KHR    = 2,  /**< Sigmoid: f(x) = 1/(1+exp(-x)). */
    VK_ML_ACTIVATION_FUNCTION_TANH_KHR       = 3,  /**< Tanh: f(x) = tanh(x). */
    VK_ML_ACTIVATION_FUNCTION_LEAKY_RELU_KHR = 4,  /**< Leaky ReLU: f(x) = x > 0 ? x : param0*x. */
    VK_ML_ACTIVATION_FUNCTION_CLAMP_KHR      = 5,  /**< Clamp: f(x) = clamp(x, param0, param1). */
    VK_ML_ACTIVATION_FUNCTION_MAX_ENUM_KHR   = 0x7FFFFFFF
} VkMLActivationFunctionKHR;

/** @brief Padding mode for convolution and pooling operations. */
typedef enum VkMLPaddingModeKHR {
    VK_ML_PADDING_MODE_VALID_KHR    = 0,  /**< No padding; output size reduced. */
    VK_ML_PADDING_MODE_SAME_KHR     = 1,  /**< Pad to preserve spatial dimensions. */
    VK_ML_PADDING_MODE_EXPLICIT_KHR = 2,  /**< Use explicit paddingTop/Bottom/Left/Right. */
    VK_ML_PADDING_MODE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkMLPaddingModeKHR;

/** @brief Logical ordering of tensor dimensions (batch, channels, spatial). */
typedef enum VkMLTensorLayoutKHR {
    VK_ML_TENSOR_LAYOUT_NCHW_KHR  = 0,  /**< Batch, Channels, Height, Width. */
    VK_ML_TENSOR_LAYOUT_NHWC_KHR  = 1,  /**< Batch, Height, Width, Channels. */
    VK_ML_TENSOR_LAYOUT_NDHWC_KHR = 2,  /**< Batch, Depth, Height, Width, Channels. */
    VK_ML_TENSOR_LAYOUT_MAX_ENUM_KHR = 0x7FFFFFFF
} VkMLTensorLayoutKHR;

/** @brief Whether a tensor binding is internal or externally-provided. */
typedef enum VkMLTensorBindingTypeKHR {
    VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR        = 0,  /**< Produced by another node; nodeIndex identifies producer. */
    VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR  = 1,  /**< Provided as graph input; tensorIndex into pExternalInputDescriptions. */
    VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR = 2,  /**< Written as graph output; tensorIndex into pExternalOutputDescriptions. */
    VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_WEIGHT_KHR = 3,  /**< Constant weight; tensorIndex into pConstantWeightDescriptions. */
    VK_ML_TENSOR_BINDING_TYPE_MAX_ENUM_KHR        = 0x7FFFFFFF
} VkMLTensorBindingTypeKHR;

/** @brief Interpolation mode for spatial resize operations. */
typedef enum VkMLResizeModeKHR {
    VK_ML_RESIZE_MODE_NEAREST_KHR  = 0,  /**< Nearest-neighbor interpolation. */
    VK_ML_RESIZE_MODE_BILINEAR_KHR = 1,  /**< Bilinear interpolation. */
    VK_ML_RESIZE_MODE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkMLResizeModeKHR;

/* ------------------------------------------------------------------ */
/* Bitmask types (T005)                                                */
/* ------------------------------------------------------------------ */

typedef VkFlags VkTensorCreateFlagsKHR;
typedef VkFlags VkTensorViewCreateFlagsKHR;
typedef VkFlags VkMLGraphCreateFlagsKHR;
typedef VkFlags VkMLSessionCreateFlagsKHR;

/* ------------------------------------------------------------------ */
/* Structure types (T007)                                              */
/* ------------------------------------------------------------------ */

/** @brief Physical device ML feature support. Chain into VkPhysicalDeviceFeatures2 or VkDeviceCreateInfo. */
typedef struct VkPhysicalDeviceMLFeaturesKHR {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           mlPrimitives;
    VkBool32           mlGraph;
    VkBool32           tensorObjects;
    VkBool32           tensorShaderAccess;
    VkBool32           tensorImageAliasing;
    VkBool32           fp16Tensors;
    VkBool32           bf16Tensors;
    VkBool32           int8Tensors;
    VkBool32           int4Tensors;
    VkBool32           fp8Tensors;
    VkBool32           fusedActivations;
    VkBool32           mlGraphScratchAutoAllocation;
} VkPhysicalDeviceMLFeaturesKHR;

/** @brief Physical device ML limits and capabilities. Chain into VkPhysicalDeviceProperties2. */
typedef struct VkPhysicalDeviceMLPropertiesKHR {
    VkStructureType    sType;
    void*              pNext;
    uint32_t           maxTensorDimensions;
    uint64_t           maxTensorElements;
    uint32_t           maxTensorDimensionSize;
    uint32_t           maxMLGraphNodeCount;
    uint32_t           maxMLGraphDepth;
    uint32_t           maxMLSessionCount;
    uint32_t           maxConcurrentMLDispatches;
    uint32_t           supportedPrimitiveCount;
    VkDeviceSize       minTensorMemoryAlignment;
    VkDeviceSize       maxScratchMemorySize;
} VkPhysicalDeviceMLPropertiesKHR;

/** @brief Describes tensor geometry, element format, memory layout, and usage. Used when creating tensors and defining graph bindings. */
typedef struct VkTensorDescriptionKHR {
    VkStructureType          sType;
    const void*              pNext;
    VkTensorTilingKHR        tiling;
    VkFormat                 format;
    uint32_t                 dimensionCount;
    const uint32_t*          pDimensions;
    const VkDeviceSize*      pStrides;
    VkTensorUsageFlagsKHR    usage;
} VkTensorDescriptionKHR;

/** @brief Parameters for creating a tensor. Pass to vkCreateTensorKHR. */
typedef struct VkTensorCreateInfoKHR {
    VkStructureType                  sType;
    const void*                      pNext;
    VkTensorCreateFlagsKHR           flags;
    const VkTensorDescriptionKHR*    pDescription;
    VkSharingMode                    sharingMode;
    uint32_t                         queueFamilyIndexCount;
    const uint32_t*                  pQueueFamilyIndices;
} VkTensorCreateInfoKHR;

/** @brief Parameters for creating a tensor view. Pass to vkCreateTensorViewKHR. Enables format reinterpretation and sub-region access. */
typedef struct VkTensorViewCreateInfoKHR {
    VkStructureType             sType;
    const void*                 pNext;
    VkTensorViewCreateFlagsKHR  flags;
    VkTensorKHR                 tensor;
    VkFormat                    format;
    uint32_t                    dimensionCount;
    const uint32_t*             pDimensionOffsets;
    const uint32_t*             pDimensionSizes;
} VkTensorViewCreateInfoKHR;

/** @brief Identifies a tensor for memory requirements query. Pass to vkGetTensorMemoryRequirementsKHR. */
typedef struct VkTensorMemoryRequirementsInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkTensorKHR        tensor;
} VkTensorMemoryRequirementsInfoKHR;

/** @brief Binds device memory to a tensor. Pass to vkBindTensorMemoryKHR. */
typedef struct VkBindTensorMemoryInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkTensorKHR        tensor;
    VkDeviceMemory     memory;
    VkDeviceSize       memoryOffset;
} VkBindTensorMemoryInfoKHR;

/** @brief Describes one copy region between tensors. Used in VkCopyTensorInfoKHR. */
typedef struct VkTensorCopyKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           dimensionCount;
    const uint32_t*    pSrcOffsets;
    const uint32_t*    pDstOffsets;
    const uint32_t*    pExtents;
} VkTensorCopyKHR;

/** @brief Parameters for a tensor copy operation. Pass to vkCmdCopyTensorKHR. */
typedef struct VkCopyTensorInfoKHR {
    VkStructureType        sType;
    const void*            pNext;
    VkTensorKHR            srcTensor;
    VkTensorKHR            dstTensor;
    uint32_t               regionCount;
    const VkTensorCopyKHR* pRegions;
} VkCopyTensorInfoKHR;

/** @brief Tensor memory barrier for synchronization. Chain into VkDependencyInfo or VkTensorDependencyInfoKHR. */
typedef struct VkTensorMemoryBarrierKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkAccessFlags2     srcAccessMask;
    VkAccessFlags2     dstAccessMask;
    uint32_t           srcQueueFamilyIndex;
    uint32_t           dstQueueFamilyIndex;
    VkTensorKHR        tensor;
} VkTensorMemoryBarrierKHR;

/** @brief Collection of tensor memory barriers. Chain into VkDependencyInfo for vkCmdPipelineBarrier2. */
typedef struct VkTensorDependencyInfoKHR {
    VkStructureType                     sType;
    const void*                         pNext;
    uint32_t                            tensorMemoryBarrierCount;
    const VkTensorMemoryBarrierKHR*     pTensorMemoryBarriers;
} VkTensorDependencyInfoKHR;

/** @brief Writes tensor views to a descriptor set. Chain into VkWriteDescriptorSet for VK_DESCRIPTOR_TYPE_TENSOR_KHR. */
typedef struct VkWriteDescriptorSetTensorKHR {
    VkStructureType          sType;
    const void*              pNext;
    uint32_t                 tensorCount;
    const VkTensorViewKHR*   pTensorViews;
} VkWriteDescriptorSetTensorKHR;

/** @brief Tensor-specific format features. Chain into VkFormatProperties3 when calling vkGetPhysicalDeviceFormatProperties2. */
typedef struct VkTensorFormatPropertiesKHR {
    VkStructureType        sType;
    void*                  pNext;
    VkFormatFeatureFlags2  tensorFeatures;
} VkTensorFormatPropertiesKHR;

/* ML Primitive Descriptors */

/** @brief Descriptor for 2D convolution and deconvolution. Use with VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR or DECONVOLUTION_2D_KHR. */
typedef struct VkMLPrimitiveDescConvolutionKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLTensorLayoutKHR          inputLayout;
    uint32_t                     kernelWidth;
    uint32_t                     kernelHeight;
    uint32_t                     strideX;
    uint32_t                     strideY;
    uint32_t                     dilationX;
    uint32_t                     dilationY;
    VkMLPaddingModeKHR           paddingMode;
    uint32_t                     paddingTop;
    uint32_t                     paddingBottom;
    uint32_t                     paddingLeft;
    uint32_t                     paddingRight;
    uint32_t                     groupCount;
    VkMLActivationFunctionKHR    fusedActivation;
    float                        activationParam0;
    float                        activationParam1;
} VkMLPrimitiveDescConvolutionKHR;

/** @brief Descriptor for GEMM: D = alpha * op(A) * op(B) + beta * C. Use with VK_ML_OPERATION_TYPE_GEMM_KHR or FULLY_CONNECTED_KHR. */
typedef struct VkMLPrimitiveDescGemmKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkBool32                     transposeA;
    VkBool32                     transposeB;
    float                        alpha;
    float                        beta;
    VkMLActivationFunctionKHR    fusedActivation;
    float                        activationParam0;
    float                        activationParam1;
} VkMLPrimitiveDescGemmKHR;

/** @brief Descriptor for max, average, or global average pooling. Use with pooling operation types. */
typedef struct VkMLPrimitiveDescPoolingKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLOperationTypeKHR         poolType;
    VkMLTensorLayoutKHR          inputLayout;
    uint32_t                     windowWidth;
    uint32_t                     windowHeight;
    uint32_t                     strideX;
    uint32_t                     strideY;
    VkMLPaddingModeKHR           paddingMode;
    uint32_t                     paddingTop;
    uint32_t                     paddingBottom;
    uint32_t                     paddingLeft;
    uint32_t                     paddingRight;
} VkMLPrimitiveDescPoolingKHR;

/** @brief Descriptor for standalone activation layer (ReLU, Sigmoid, Tanh, Leaky ReLU, Clamp). */
typedef struct VkMLPrimitiveDescActivationKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLActivationFunctionKHR    activationType;
    float                        param0;
    float                        param1;
} VkMLPrimitiveDescActivationKHR;

/** @brief Descriptor for batch normalization or LRN. Use with VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR or LRN_KHR. */
typedef struct VkMLPrimitiveDescNormalizationKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLOperationTypeKHR         normType;
    float                        epsilon;
    VkMLTensorLayoutKHR          inputLayout;
    VkMLActivationFunctionKHR    fusedActivation;
    float                        activationParam0;
    float                        activationParam1;
} VkMLPrimitiveDescNormalizationKHR;

/** @brief Descriptor for element-wise add or multiply. Use with VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR or ELEMENTWISE_MUL_KHR. */
typedef struct VkMLPrimitiveDescElementwiseKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLOperationTypeKHR         opType;
    VkMLActivationFunctionKHR    fusedActivation;
    float                        activationParam0;
    float                        activationParam1;
} VkMLPrimitiveDescElementwiseKHR;

/** @brief Descriptor for concatenation along a specified dimension. */
typedef struct VkMLPrimitiveDescConcatKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           axis;        /**< Dimension index along which to concatenate. */
} VkMLPrimitiveDescConcatKHR;

/** @brief Descriptor for tensor reshape (reinterpret dimensions without copying data). */
typedef struct VkMLPrimitiveDescReshapeKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           dimensionCount;      /**< Number of output dimensions. */
    const uint32_t*    pOutputDimensions;    /**< Desired output shape. */
} VkMLPrimitiveDescReshapeKHR;

/** @brief Descriptor for dimension transposition. */
typedef struct VkMLPrimitiveDescTransposeKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           dimensionCount;   /**< Number of dimensions. */
    const uint32_t*    pPermutation;     /**< Permutation indices; pPermutation[i] is the source dim for output dim i. */
} VkMLPrimitiveDescTransposeKHR;

/** @brief Descriptor for spatial resize (upsampling/downsampling). */
typedef struct VkMLPrimitiveDescResizeKHR {
    VkStructureType     sType;
    const void*         pNext;
    VkMLResizeModeKHR   mode;           /**< Interpolation mode. */
    float               scaleHeight;    /**< Vertical scale factor; must be > 0. */
    float               scaleWidth;     /**< Horizontal scale factor; must be > 0. */
} VkMLPrimitiveDescResizeKHR;

/* ML Graph Structures */

/** @brief Binds a tensor to a graph node input or output. Used in VkMLGraphNodeCreateInfoKHR. */
typedef struct VkMLTensorBindingKHR {
    VkStructureType                sType;
    const void*                    pNext;
    VkMLTensorBindingTypeKHR       bindingType;
    uint32_t                       nodeIndex;
    uint32_t                       tensorIndex;
    const VkTensorDescriptionKHR*  pTensorDescription;
} VkMLTensorBindingKHR;

/** @brief Defines one node in an ML graph. Specifies operation type, descriptor, and input/output bindings. */
typedef struct VkMLGraphNodeCreateInfoKHR {
    VkStructureType             sType;
    const void*                 pNext;
    VkMLOperationTypeKHR        operationType;
    const void*                 pOperationDesc;
    uint32_t                    inputCount;
    const VkMLTensorBindingKHR* pInputs;
    uint32_t                    outputCount;
    const VkMLTensorBindingKHR* pOutputs;
    const char*                 pNodeName;
} VkMLGraphNodeCreateInfoKHR;

/** @brief Parameters for creating an ML graph. Pass to vkCreateMLGraphKHR. Defines nodes and external inputs/outputs/weights. */
typedef struct VkMLGraphCreateInfoKHR {
    VkStructureType                      sType;
    const void*                          pNext;
    VkMLGraphCreateFlagsKHR              flags;
    uint32_t                             nodeCount;
    const VkMLGraphNodeCreateInfoKHR*    pNodes;
    uint32_t                             externalInputCount;
    const VkTensorDescriptionKHR*        pExternalInputDescriptions;
    uint32_t                             externalOutputCount;
    const VkTensorDescriptionKHR*        pExternalOutputDescriptions;
    uint32_t                             constantWeightCount;
    const VkTensorDescriptionKHR*        pConstantWeightDescriptions;
} VkMLGraphCreateInfoKHR;

/* ML Session and Dispatch Structures */

/** @brief Parameters for creating an ML session. Pass to vkCreateMLSessionKHR. Associates graph with scratch memory. */
typedef struct VkMLSessionCreateInfoKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLSessionCreateFlagsKHR    flags;
    VkMLGraphKHR                 graph;
    VkDeviceMemory               scratchMemory;
    VkDeviceSize                 scratchMemoryOffset;
    VkDeviceSize                 scratchMemorySize;
} VkMLSessionCreateInfoKHR;

/** @brief Parameters for dispatching an ML graph. Pass to vkCmdDispatchMLGraphKHR. Binds input, output, and weight tensors. */
typedef struct VkMLGraphDispatchInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkMLSessionKHR     session;
    uint32_t           inputTensorCount;
    const VkTensorKHR* pInputTensors;
    uint32_t           outputTensorCount;
    const VkTensorKHR* pOutputTensors;
    uint32_t           weightTensorCount;
    const VkTensorKHR* pWeightTensors;
} VkMLGraphDispatchInfoKHR;

/* ------------------------------------------------------------------ */
/* Function prototypes (T008)                                          */
/* ------------------------------------------------------------------ */

/* Tensor lifecycle */

/**
 * @brief Create a tensor object.
 * @param device The logical device that creates the tensor.
 * @param pCreateInfo Pointer to VkTensorCreateInfoKHR describing the tensor.
 * @param pAllocator Controls host memory allocation; may be NULL.
 * @param pTensor Pointer to a VkTensorKHR handle in which the result is returned.
 * @return VK_SUCCESS, or VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_VALIDATION_FAILED.
 * @note Requires tensorObjects feature to be enabled.
 */
VKAPI_ATTR VkResult VKAPI_CALL vkCreateTensorKHR(
    VkDevice                        device,
    const VkTensorCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*    pAllocator,
    VkTensorKHR*                    pTensor);

/**
 * @brief Destroy a tensor object.
 * @param device The logical device that owns the tensor.
 * @param tensor The tensor to destroy.
 * @param pAllocator Controls host memory deallocation; must match allocation used at creation.
 * @note All submitted commands that refer to tensor must have completed. All VkTensorViewKHR from tensor must be destroyed first.
 */
VKAPI_ATTR void VKAPI_CALL vkDestroyTensorKHR(
    VkDevice                        device,
    VkTensorKHR                     tensor,
    const VkAllocationCallbacks*    pAllocator);

/**
 * @brief Create a tensor view.
 * @param device The logical device that creates the view.
 * @param pCreateInfo Pointer to VkTensorViewCreateInfoKHR describing the view.
 * @param pAllocator Controls host memory allocation; may be NULL.
 * @param pView Pointer to a VkTensorViewKHR handle in which the result is returned.
 * @return VK_SUCCESS, or VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY.
 * @note The source tensor must have memory bound. Requires tensorObjects feature.
 */
VKAPI_ATTR VkResult VKAPI_CALL vkCreateTensorViewKHR(
    VkDevice                            device,
    const VkTensorViewCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*        pAllocator,
    VkTensorViewKHR*                    pView);

/**
 * @brief Destroy a tensor view.
 * @param device The logical device that owns the view.
 * @param view The tensor view to destroy.
 * @param pAllocator Controls host memory deallocation; must match allocation used at creation.
 */
VKAPI_ATTR void VKAPI_CALL vkDestroyTensorViewKHR(
    VkDevice                        device,
    VkTensorViewKHR                 view,
    const VkAllocationCallbacks*    pAllocator);

/* Tensor memory */

/**
 * @brief Query memory requirements for a tensor.
 * @param device The logical device that owns the tensor.
 * @param pInfo Pointer to VkTensorMemoryRequirementsInfoKHR identifying the tensor.
 * @param pMemoryRequirements Pointer to VkMemoryRequirements2 in which requirements are returned.
 */
VKAPI_ATTR void VKAPI_CALL vkGetTensorMemoryRequirementsKHR(
    VkDevice                                    device,
    const VkTensorMemoryRequirementsInfoKHR*     pInfo,
    VkMemoryRequirements2*                       pMemoryRequirements);

/**
 * @brief Bind device memory to one or more tensors.
 * @param device The logical device that owns the tensors.
 * @param bindInfoCount Number of elements in pBindInfos.
 * @param pBindInfos Pointer to array of VkBindTensorMemoryInfoKHR structures.
 * @return VK_SUCCESS, or VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY.
 * @note Each tensor must not already be bound. Memory must satisfy requirements from vkGetTensorMemoryRequirementsKHR.
 */
VKAPI_ATTR VkResult VKAPI_CALL vkBindTensorMemoryKHR(
    VkDevice                              device,
    uint32_t                              bindInfoCount,
    const VkBindTensorMemoryInfoKHR*      pBindInfos);

/* Tensor commands */

/**
 * @brief Copy data between tensors.
 * @param commandBuffer The command buffer into which the command is recorded.
 * @param pCopyInfo Pointer to VkCopyTensorInfoKHR describing the copy operation.
 * @note Must be called outside a render pass. Source and destination must have TRANSFER_SRC and TRANSFER_DST usage respectively.
 */
VKAPI_ATTR void VKAPI_CALL vkCmdCopyTensorKHR(
    VkCommandBuffer                 commandBuffer,
    const VkCopyTensorInfoKHR*      pCopyInfo);

/* ML graph lifecycle */

/**
 * @brief Create an ML graph.
 * @param device The logical device that creates the graph.
 * @param pCreateInfo Pointer to VkMLGraphCreateInfoKHR describing the graph.
 * @param pAllocator Controls host memory allocation; may be NULL.
 * @param pGraph Pointer to a VkMLGraphKHR handle in which the result is returned.
 * @return VK_SUCCESS, or VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_INITIALIZATION_FAILED.
 * @note Requires mlGraph feature. Graph is immutable once created.
 */
VKAPI_ATTR VkResult VKAPI_CALL vkCreateMLGraphKHR(
    VkDevice                         device,
    const VkMLGraphCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*     pAllocator,
    VkMLGraphKHR*                    pGraph);

/**
 * @brief Destroy an ML graph.
 * @param device The logical device that owns the graph.
 * @param graph The ML graph to destroy.
 * @param pAllocator Controls host memory deallocation; must match allocation used at creation.
 * @note All submitted commands referencing graph must have completed. All VkMLSessionKHR must be destroyed first.
 */
VKAPI_ATTR void VKAPI_CALL vkDestroyMLGraphKHR(
    VkDevice                        device,
    VkMLGraphKHR                    graph,
    const VkAllocationCallbacks*    pAllocator);

/**
 * @brief Query scratch memory requirements for an ML graph.
 * @param device The logical device that owns the graph.
 * @param graph The ML graph to query.
 * @param pMemoryRequirements Pointer to VkMemoryRequirements2 in which requirements are returned.
 * @note If mlGraphScratchAutoAllocation is enabled, scratch may be allocated internally; otherwise bind via session.
 */
VKAPI_ATTR void VKAPI_CALL vkGetMLGraphMemoryRequirementsKHR(
    VkDevice                device,
    VkMLGraphKHR            graph,
    VkMemoryRequirements2*  pMemoryRequirements);

/* ML session lifecycle */

/**
 * @brief Create an ML session.
 * @param device The logical device that creates the session.
 * @param pCreateInfo Pointer to VkMLSessionCreateInfoKHR describing the session.
 * @param pAllocator Controls host memory allocation; may be NULL.
 * @param pSession Pointer to a VkMLSessionKHR handle in which the result is returned.
 * @return VK_SUCCESS, or VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY.
 * @note Scratch memory size must satisfy vkGetMLGraphMemoryRequirementsKHR. If scratchMemory is VK_NULL_HANDLE, mlGraphScratchAutoAllocation must be enabled.
 */
VKAPI_ATTR VkResult VKAPI_CALL vkCreateMLSessionKHR(
    VkDevice                           device,
    const VkMLSessionCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*       pAllocator,
    VkMLSessionKHR*                    pSession);

/**
 * @brief Destroy an ML session.
 * @param device The logical device that owns the session.
 * @param session The ML session to destroy.
 * @param pAllocator Controls host memory deallocation; must match allocation used at creation.
 * @note All submitted commands referencing session must have completed.
 */
VKAPI_ATTR void VKAPI_CALL vkDestroyMLSessionKHR(
    VkDevice                        device,
    VkMLSessionKHR                  session,
    const VkAllocationCallbacks*    pAllocator);

/* ML dispatch */

/**
 * @brief Record an ML graph dispatch command.
 * @param commandBuffer The command buffer into which the command is recorded.
 * @param pDispatchInfo Pointer to VkMLGraphDispatchInfoKHR describing the dispatch.
 * @note Command buffer must be in recording state. Input/output/weight counts must match the graph's external input/output/constant weight counts.
 */
VKAPI_ATTR void VKAPI_CALL vkCmdDispatchMLGraphKHR(
    VkCommandBuffer                     commandBuffer,
    const VkMLGraphDispatchInfoKHR*     pDispatchInfo);

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_ML_PRIMITIVES_H_ */
