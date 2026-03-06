# Quickstart: VK_KHR_ml_primitives

**Branch**: `001-ml-primitives` | **Date**: 2026-03-05

This guide walks through building and running a minimal ML inference
pipeline using the `VK_KHR_ml_primitives` extension.

---

## Prerequisites

- Vulkan 1.3 SDK installed
- CMake 3.20+
- C11-capable compiler (GCC 11+, Clang 14+, or MSVC 2022+)
- A Vulkan 1.3 driver with `VK_KHR_ml_primitives` support

## Build

```bash
git clone <repo-url> vulkan-ml-api
cd vulkan-ml-api
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Run Tests

```bash
cd build
ctest --output-on-failure
```

## Minimal Example: Single-Layer Convolution

The following C program creates a tensor, builds a single-node
convolution graph, and dispatches it. Error checking is abbreviated.

### Step 1: Query ML Support

```c
VkPhysicalDeviceMLFeaturesKHR mlFeatures = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
};
VkPhysicalDeviceFeatures2 features2 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    .pNext = &mlFeatures,
};
vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

if (!mlFeatures.tensorObjects || !mlFeatures.mlPrimitives ||
    !mlFeatures.mlGraph) {
    /* ML not supported on this device */
    return;
}
```

### Step 2: Create an Input Tensor

```c
const uint32_t inputDims[] = {1, 3, 224, 224};
VkTensorDescriptionKHR inputDesc = {
    .sType          = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
    .tiling         = VK_TENSOR_TILING_OPTIMAL_KHR,
    .format         = VK_FORMAT_R16_SFLOAT,
    .dimensionCount = 4,
    .pDimensions    = inputDims,
    .pStrides       = NULL,
    .usage          = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR |
                      VK_TENSOR_USAGE_TRANSFER_DST_BIT_KHR,
};

VkTensorCreateInfoKHR inputCI = {
    .sType        = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
    .pDescription = &inputDesc,
    .sharingMode  = VK_SHARING_MODE_EXCLUSIVE,
};

VkTensorKHR inputTensor;
vkCreateTensorKHR(device, &inputCI, NULL, &inputTensor);
```

### Step 3: Query Memory and Bind

```c
VkTensorMemoryRequirementsInfoKHR memReqInfo = {
    .sType  = VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_KHR,
    .tensor = inputTensor,
};
VkMemoryRequirements2 memReqs = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
};
vkGetTensorMemoryRequirementsKHR(device, &memReqInfo, &memReqs);

/* Allocate and bind (select appropriate memory type) */
VkDeviceMemory inputMemory;
/* ... vkAllocateMemory with memReqs.memoryRequirements ... */

VkBindTensorMemoryInfoKHR bindInfo = {
    .sType        = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
    .tensor       = inputTensor,
    .memory       = inputMemory,
    .memoryOffset = 0,
};
vkBindTensorMemoryKHR(device, 1, &bindInfo);
```

### Step 4: Build an ML Graph (Single Convolution)

```c
VkMLPrimitiveDescConvolutionKHR convDesc = {
    .sType           = VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
    .inputLayout     = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
    .kernelWidth     = 3,
    .kernelHeight    = 3,
    .strideX         = 1,
    .strideY         = 1,
    .dilationX       = 1,
    .dilationY       = 1,
    .paddingMode     = VK_ML_PADDING_MODE_SAME_KHR,
    .groupCount      = 1,
    .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
};

/* Define tensor bindings for the graph node */
VkMLTensorBindingKHR nodeInputBinding = {
    .sType              = VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
    .bindingType        = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR,
    .tensorIndex        = 0,
    .pTensorDescription = &inputDesc,
};
VkMLTensorBindingKHR nodeWeightBinding = {
    .sType              = VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
    .bindingType        = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_WEIGHT_KHR,
    .tensorIndex        = 0,
    .pTensorDescription = &weightDesc,  /* defined similarly */
};
const VkMLTensorBindingKHR nodeInputs[] = {
    nodeInputBinding, nodeWeightBinding
};

VkMLTensorBindingKHR nodeOutputBinding = {
    .sType              = VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
    .bindingType        = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
    .tensorIndex        = 0,
    .pTensorDescription = &outputDesc,  /* defined similarly */
};

VkMLGraphNodeCreateInfoKHR node = {
    .sType          = VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
    .operationType  = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
    .pOperationDesc = &convDesc,
    .inputCount     = 2,
    .pInputs        = nodeInputs,
    .outputCount    = 1,
    .pOutputs       = &nodeOutputBinding,
    .pNodeName      = "conv2d",
};

VkMLGraphCreateInfoKHR graphCI = {
    .sType                       = VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
    .nodeCount                   = 1,
    .pNodes                      = &node,
    .externalInputCount          = 1,
    .pExternalInputDescriptions  = &inputDesc,
    .externalOutputCount         = 1,
    .pExternalOutputDescriptions = &outputDesc,
    .constantWeightCount         = 1,
    .pConstantWeightDescriptions = &weightDesc,
};

VkMLGraphKHR mlGraph;
vkCreateMLGraphKHR(device, &graphCI, NULL, &mlGraph);
```

### Step 5: Create Session and Dispatch

```c
/* Query scratch memory */
VkMemoryRequirements2 scratchReqs = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
};
vkGetMLGraphMemoryRequirementsKHR(device, mlGraph, &scratchReqs);

/* Allocate scratch and create session */
VkDeviceMemory scratchMemory;
/* ... vkAllocateMemory ... */

VkMLSessionCreateInfoKHR sessionCI = {
    .sType               = VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
    .graph               = mlGraph,
    .scratchMemory       = scratchMemory,
    .scratchMemoryOffset = 0,
    .scratchMemorySize   = scratchReqs.memoryRequirements.size,
};

VkMLSessionKHR session;
vkCreateMLSessionKHR(device, &sessionCI, NULL, &session);

/* Record dispatch into a command buffer */
VkMLGraphDispatchInfoKHR dispatchInfo = {
    .sType             = VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
    .session           = session,
    .inputTensorCount  = 1,
    .pInputTensors     = &inputTensor,
    .outputTensorCount = 1,
    .pOutputTensors    = &outputTensor,
    .weightTensorCount = 1,
    .pWeightTensors    = &weightTensor,
};

vkCmdDispatchMLGraphKHR(cmdBuf, &dispatchInfo);
```

### Step 6: Synchronize and Read Results

```c
/* Barrier: ML write → transfer read */
VkTensorMemoryBarrierKHR barrier = {
    .sType               = VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
    .srcAccessMask       = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
    .dstAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .tensor              = outputTensor,
};
VkTensorDependencyInfoKHR tensorDepInfo = {
    .sType                    = VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR,
    .tensorMemoryBarrierCount = 1,
    .pTensorMemoryBarriers    = &barrier,
};
VkDependencyInfo depInfo = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .pNext = &tensorDepInfo,
};
vkCmdPipelineBarrier2(cmdBuf, &depInfo);

/* Submit and wait, then read output tensor data */
```

### Step 7: Cleanup

```c
vkDestroyMLSessionKHR(device, session, NULL);
vkDestroyMLGraphKHR(device, mlGraph, NULL);
vkDestroyTensorKHR(device, outputTensor, NULL);
vkDestroyTensorKHR(device, weightTensor, NULL);
vkDestroyTensorKHR(device, inputTensor, NULL);
vkFreeMemory(device, scratchMemory, NULL);
/* ... free other memory ... */
```

---

## Validation

Run with the validation layer enabled to catch VUID violations:

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHR_ml_validation ./your_app
```

---

## Next Steps

- See `spec/VK_KHR_ml_primitives.adoc` for the full specification
- See `specs/001-ml-primitives/data-model.md` for entity relationships
- See `specs/001-ml-primitives/contracts/c-api.md` for the complete API surface
- Run `/speckit.tasks` to generate the implementation task breakdown
