# C API Contract: VK_KHR_ml_primitives

**Branch**: `001-ml-primitives` | **Date**: 2026-03-05

This document defines the public C API surface of the extension.
All types, enumerations, and function prototypes below MUST appear in
`include/vulkan/vulkan_ml_primitives.h`. The API contract is derived
directly from `spec/VK_KHR_ml_primitives.adoc`.

---

## Handles

```c
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkTensorKHR)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkTensorViewKHR)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkMLGraphKHR)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkMLSessionKHR)
```

---

## Enumerations

```c
typedef enum VkTensorTilingKHR {
    VK_TENSOR_TILING_OPTIMAL_KHR = 0,
    VK_TENSOR_TILING_LINEAR_KHR  = 1,
} VkTensorTilingKHR;

typedef enum VkTensorUsageFlagBitsKHR {
    VK_TENSOR_USAGE_SHADER_BIT_KHR          = 0x00000001,
    VK_TENSOR_USAGE_TRANSFER_SRC_BIT_KHR    = 0x00000002,
    VK_TENSOR_USAGE_TRANSFER_DST_BIT_KHR    = 0x00000004,
    VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR  = 0x00000008,
    VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR = 0x00000010,
    VK_TENSOR_USAGE_ML_GRAPH_WEIGHT_BIT_KHR = 0x00000020,
    VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_KHR  = 0x00000040,
} VkTensorUsageFlagBitsKHR;
typedef VkFlags VkTensorUsageFlagsKHR;

typedef enum VkMLOperationTypeKHR {
    VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR      = 0,
    VK_ML_OPERATION_TYPE_DECONVOLUTION_2D_KHR    = 1,
    VK_ML_OPERATION_TYPE_GEMM_KHR                = 2,
    VK_ML_OPERATION_TYPE_FULLY_CONNECTED_KHR     = 3,
    VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR         = 4,
    VK_ML_OPERATION_TYPE_AVERAGE_POOL_2D_KHR     = 5,
    VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR = 6,
    VK_ML_OPERATION_TYPE_RELU_KHR                = 7,
    VK_ML_OPERATION_TYPE_SIGMOID_KHR             = 8,
    VK_ML_OPERATION_TYPE_TANH_KHR                = 9,
    VK_ML_OPERATION_TYPE_LEAKY_RELU_KHR          = 10,
    VK_ML_OPERATION_TYPE_PRELU_KHR               = 11,
    VK_ML_OPERATION_TYPE_SOFTMAX_KHR             = 12,
    VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR = 13,
    VK_ML_OPERATION_TYPE_LRN_KHR                 = 14,
    VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR     = 15,
    VK_ML_OPERATION_TYPE_ELEMENTWISE_MUL_KHR     = 16,
    VK_ML_OPERATION_TYPE_CONCAT_KHR              = 17,
    VK_ML_OPERATION_TYPE_RESHAPE_KHR             = 18,
    VK_ML_OPERATION_TYPE_TRANSPOSE_KHR           = 19,
    VK_ML_OPERATION_TYPE_RESIZE_KHR              = 20,
} VkMLOperationTypeKHR;

typedef enum VkMLActivationFunctionKHR {
    VK_ML_ACTIVATION_FUNCTION_NONE_KHR       = 0,
    VK_ML_ACTIVATION_FUNCTION_RELU_KHR       = 1,
    VK_ML_ACTIVATION_FUNCTION_SIGMOID_KHR    = 2,
    VK_ML_ACTIVATION_FUNCTION_TANH_KHR       = 3,
    VK_ML_ACTIVATION_FUNCTION_LEAKY_RELU_KHR = 4,
    VK_ML_ACTIVATION_FUNCTION_CLAMP_KHR      = 5,
} VkMLActivationFunctionKHR;

typedef enum VkMLPaddingModeKHR {
    VK_ML_PADDING_MODE_VALID_KHR    = 0,
    VK_ML_PADDING_MODE_SAME_KHR     = 1,
    VK_ML_PADDING_MODE_EXPLICIT_KHR = 2,
} VkMLPaddingModeKHR;

typedef enum VkMLTensorLayoutKHR {
    VK_ML_TENSOR_LAYOUT_NCHW_KHR  = 0,
    VK_ML_TENSOR_LAYOUT_NHWC_KHR  = 1,
    VK_ML_TENSOR_LAYOUT_NDHWC_KHR = 2,
} VkMLTensorLayoutKHR;

typedef enum VkMLTensorBindingTypeKHR {
    VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR        = 0,
    VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR  = 1,
    VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR = 2,
    VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_WEIGHT_KHR = 3,
} VkMLTensorBindingTypeKHR;
```

---

## Bitmask Types

```c
typedef VkFlags VkTensorCreateFlagsKHR;
typedef VkFlags VkTensorViewCreateFlagsKHR;
typedef VkFlags VkMLGraphCreateFlagsKHR;
typedef VkFlags VkMLSessionCreateFlagsKHR;
```

---

## Structures

### Feature and Property Queries

```c
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
```

### Tensor Structures

```c
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

typedef struct VkTensorCreateInfoKHR {
    VkStructureType                  sType;
    const void*                      pNext;
    VkTensorCreateFlagsKHR           flags;
    const VkTensorDescriptionKHR*    pDescription;
    VkSharingMode                    sharingMode;
    uint32_t                         queueFamilyIndexCount;
    const uint32_t*                  pQueueFamilyIndices;
} VkTensorCreateInfoKHR;

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

typedef struct VkTensorMemoryRequirementsInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkTensorKHR        tensor;
} VkTensorMemoryRequirementsInfoKHR;

typedef struct VkBindTensorMemoryInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkTensorKHR        tensor;
    VkDeviceMemory     memory;
    VkDeviceSize       memoryOffset;
} VkBindTensorMemoryInfoKHR;
```

### Tensor Copy Structures

```c
typedef struct VkTensorCopyKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           dimensionCount;
    const uint32_t*    pSrcOffsets;
    const uint32_t*    pDstOffsets;
    const uint32_t*    pExtents;
} VkTensorCopyKHR;

typedef struct VkCopyTensorInfoKHR {
    VkStructureType        sType;
    const void*            pNext;
    VkTensorKHR            srcTensor;
    VkTensorKHR            dstTensor;
    uint32_t               regionCount;
    const VkTensorCopyKHR* pRegions;
} VkCopyTensorInfoKHR;
```

### Tensor Barrier Structures

```c
typedef struct VkTensorMemoryBarrierKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkAccessFlags2     srcAccessMask;
    VkAccessFlags2     dstAccessMask;
    uint32_t           srcQueueFamilyIndex;
    uint32_t           dstQueueFamilyIndex;
    VkTensorKHR        tensor;
} VkTensorMemoryBarrierKHR;

typedef struct VkTensorDependencyInfoKHR {
    VkStructureType                     sType;
    const void*                         pNext;
    uint32_t                            tensorMemoryBarrierCount;
    const VkTensorMemoryBarrierKHR*     pTensorMemoryBarriers;
} VkTensorDependencyInfoKHR;
```

### Tensor Descriptor Structures

```c
typedef struct VkWriteDescriptorSetTensorKHR {
    VkStructureType          sType;
    const void*              pNext;
    uint32_t                 tensorCount;
    const VkTensorViewKHR*   pTensorViews;
} VkWriteDescriptorSetTensorKHR;

typedef struct VkTensorFormatPropertiesKHR {
    VkStructureType        sType;
    void*                  pNext;
    VkFormatFeatureFlags2  tensorFeatures;
} VkTensorFormatPropertiesKHR;
```

### ML Primitive Descriptors

```c
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

typedef struct VkMLPrimitiveDescActivationKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLActivationFunctionKHR    activationType;
    float                        param0;
    float                        param1;
} VkMLPrimitiveDescActivationKHR;

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

typedef struct VkMLPrimitiveDescElementwiseKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLOperationTypeKHR         opType;
    VkMLActivationFunctionKHR    fusedActivation;
    float                        activationParam0;
    float                        activationParam1;
} VkMLPrimitiveDescElementwiseKHR;
```

### ML Graph Structures

```c
typedef struct VkMLTensorBindingKHR {
    VkStructureType                sType;
    const void*                    pNext;
    VkMLTensorBindingTypeKHR       bindingType;
    uint32_t                       nodeIndex;
    uint32_t                       tensorIndex;
    const VkTensorDescriptionKHR*  pTensorDescription;
} VkMLTensorBindingKHR;

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
```

### ML Session and Dispatch Structures

```c
typedef struct VkMLSessionCreateInfoKHR {
    VkStructureType              sType;
    const void*                  pNext;
    VkMLSessionCreateFlagsKHR    flags;
    VkMLGraphKHR                 graph;
    VkDeviceMemory               scratchMemory;
    VkDeviceSize                 scratchMemoryOffset;
    VkDeviceSize                 scratchMemorySize;
} VkMLSessionCreateInfoKHR;

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
```

---

## Function Prototypes

### Tensor Lifecycle

```c
VkResult vkCreateTensorKHR(
    VkDevice                        device,
    const VkTensorCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*    pAllocator,
    VkTensorKHR*                    pTensor);

void vkDestroyTensorKHR(
    VkDevice                        device,
    VkTensorKHR                     tensor,
    const VkAllocationCallbacks*    pAllocator);

VkResult vkCreateTensorViewKHR(
    VkDevice                            device,
    const VkTensorViewCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*        pAllocator,
    VkTensorViewKHR*                    pView);

void vkDestroyTensorViewKHR(
    VkDevice                        device,
    VkTensorViewKHR                 view,
    const VkAllocationCallbacks*    pAllocator);
```

### Tensor Memory

```c
void vkGetTensorMemoryRequirementsKHR(
    VkDevice                                    device,
    const VkTensorMemoryRequirementsInfoKHR*     pInfo,
    VkMemoryRequirements2*                       pMemoryRequirements);

VkResult vkBindTensorMemoryKHR(
    VkDevice                              device,
    uint32_t                              bindInfoCount,
    const VkBindTensorMemoryInfoKHR*      pBindInfos);
```

### Tensor Commands

```c
void vkCmdCopyTensorKHR(
    VkCommandBuffer                 commandBuffer,
    const VkCopyTensorInfoKHR*      pCopyInfo);
```

### ML Graph Lifecycle

```c
VkResult vkCreateMLGraphKHR(
    VkDevice                         device,
    const VkMLGraphCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*     pAllocator,
    VkMLGraphKHR*                    pGraph);

void vkDestroyMLGraphKHR(
    VkDevice                        device,
    VkMLGraphKHR                    graph,
    const VkAllocationCallbacks*    pAllocator);

void vkGetMLGraphMemoryRequirementsKHR(
    VkDevice                device,
    VkMLGraphKHR            graph,
    VkMemoryRequirements2*  pMemoryRequirements);
```

### ML Session Lifecycle

```c
VkResult vkCreateMLSessionKHR(
    VkDevice                           device,
    const VkMLSessionCreateInfoKHR*    pCreateInfo,
    const VkAllocationCallbacks*       pAllocator,
    VkMLSessionKHR*                    pSession);

void vkDestroyMLSessionKHR(
    VkDevice                        device,
    VkMLSessionKHR                  session,
    const VkAllocationCallbacks*    pAllocator);
```

### ML Dispatch

```c
void vkCmdDispatchMLGraphKHR(
    VkCommandBuffer                     commandBuffer,
    const VkMLGraphDispatchInfoKHR*     pDispatchInfo);
```

---

## Return Codes

| Function | Success | Failure |
|----------|---------|---------|
| `vkCreateTensorKHR` | `VK_SUCCESS` | `VK_ERROR_OUT_OF_HOST_MEMORY`, `VK_ERROR_OUT_OF_DEVICE_MEMORY`, `VK_ERROR_VALIDATION_FAILED` |
| `vkBindTensorMemoryKHR` | `VK_SUCCESS` | `VK_ERROR_OUT_OF_HOST_MEMORY`, `VK_ERROR_OUT_OF_DEVICE_MEMORY` |
| `vkCreateTensorViewKHR` | `VK_SUCCESS` | `VK_ERROR_OUT_OF_HOST_MEMORY`, `VK_ERROR_OUT_OF_DEVICE_MEMORY` |
| `vkCreateMLGraphKHR` | `VK_SUCCESS` | `VK_ERROR_OUT_OF_HOST_MEMORY`, `VK_ERROR_OUT_OF_DEVICE_MEMORY`, `VK_ERROR_INITIALIZATION_FAILED` |
| `vkCreateMLSessionKHR` | `VK_SUCCESS` | `VK_ERROR_OUT_OF_HOST_MEMORY`, `VK_ERROR_OUT_OF_DEVICE_MEMORY` |

---

## Extension Constants

```c
#define VK_KHR_ML_PRIMITIVES_SPEC_VERSION 1
#define VK_KHR_ML_PRIMITIVES_EXTENSION_NAME "VK_KHR_ml_primitives"
```
