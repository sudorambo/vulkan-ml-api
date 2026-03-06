# Data Model: VK_KHR_ml_primitives

**Plan**: [plan.md](plan.md) | **Date**: 2026-03-05

## Entity Overview

The extension defines 4 Vulkan object types, 10 primitive descriptor types,
and supporting structures for synchronization, dispatch, and configuration.

```
VkTensorKHR ──────────┐
    │                  │
    ▼                  │ bound via VkBindTensorMemoryInfoKHR
VkTensorViewKHR        │
    │                  ▼
    │            VkDeviceMemory
    │
    ├── used as input/output/weight in ──▶ VkMLGraphDispatchInfoKHR
    │
    ▼
VkMLGraphKHR ────────────────────────────▶ VkMLSessionKHR
    │                                          │
    ├── nodeCount × VkMLGraphNodeCreateInfoKHR  │
    │       │                                   │
    │       ├── VkMLPrimitiveDesc*KHR           │
    │       └── VkMLTensorBindingKHR[]          │
    │                                           │
    ├── externalInputDescs[]                    │
    ├── externalOutputDescs[]                   ▼
    └── constantWeightDescs[]            vkCmdDispatchMLGraphKHR
```

## Entities

### VkTensorKHR (Non-dispatchable handle)

N-dimensional data container for ML workloads.

| Field | Type | Constraints |
|-------|------|-------------|
| description | VkTensorDescriptionKHR | Copied at creation |
| dimensions | uint32_t[] | Deep-copied; count = description.dimensionCount |
| strides | VkDeviceSize[] | Deep-copied; NULL for optimal tiling |
| sharingMode | VkSharingMode | EXCLUSIVE or CONCURRENT |
| queueFamilyIndexCount | uint32_t | >= 2 when CONCURRENT |
| queueFamilyIndices | uint32_t[] | Deep-copied |
| boundMemory | VkDeviceMemory | VK_NULL_HANDLE until bound |
| memoryOffset | VkDeviceSize | 0 until bound |
| memoryBound | VkBool32 | VK_FALSE until bound; immutable after |

**Lifecycle**: Create → Query memory → Allocate → Bind → Use → Destroy

**Validation rules**:
- dimensionCount: 1–`maxTensorDimensions` (VUID_TENSOR_DESC_DIM_COUNT)
- Each dimension: 1–`maxTensorDimensionSize` (VUID_TENSOR_DESC_DIM_VALUES)
- Product of dimensions: ≤ `maxTensorElements` (VUID_TENSOR_DESC_DIM_PRODUCT)
- Strides must be NULL for optimal tiling (VUID_TENSOR_DESC_STRIDES_OPTIMAL)
- Strides must be aligned to element size (VUID_TENSOR_DESC_STRIDE_ALIGN)
- Format must be supported (VUID_TENSOR_DESC_FORMAT)
- Usage must be non-zero and within valid mask
- sType must be VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR

### VkTensorViewKHR (Non-dispatchable handle)

Typed sub-region view into a VkTensorKHR.

| Field | Type | Constraints |
|-------|------|-------------|
| tensor | VkTensorKHR | Parent; must have memory bound |
| format | VkFormat | Same element size as parent |
| dimensionCount | uint32_t | Must match parent |
| dimensionOffsets | uint32_t[] | Deep-copied |
| dimensionSizes | uint32_t[] | Deep-copied |

**Lifecycle**: Create (parent must have memory bound) → Use → Destroy (before parent)

**Validation rules**:
- Parent tensor must have memory bound (VUID_TENSOR_VIEW_MEMORY_BOUND)
- Format element size must match parent (VUID_TENSOR_VIEW_FORMAT)
- dimensionCount must match parent (VUID_TENSOR_VIEW_DIM_COUNT)
- offset[i] + size[i] ≤ parentDim[i] for all i, **overflow-safe** (VUID_TENSOR_VIEW_RANGE)
- sType must be VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR

### VkMLGraphKHR (Non-dispatchable handle)

Compiled DAG of ML primitive operations. Immutable after creation.

| Field | Type | Constraints |
|-------|------|-------------|
| nodeCount | uint32_t | 1–`maxMLGraphNodeCount` |
| nodes | VkMLGraphNodeCreateInfoKHR[] | Deep-copied with all pointer members |
| externalInputCount | uint32_t | ≥ 1 |
| externalInputDescs | VkTensorDescriptionKHR[] | Deep-copied |
| externalOutputCount | uint32_t | ≥ 1 |
| externalOutputDescs | VkTensorDescriptionKHR[] | Deep-copied |
| constantWeightCount | uint32_t | ≥ 0 |
| constantWeightDescs | VkTensorDescriptionKHR[] | Deep-copied |
| scratchMemorySize | VkDeviceSize | Computed from tensor desc sizes |

**Lifecycle**: Create → Query scratch requirements → Create session(s) → Destroy (after all sessions)

**Validation rules**:
- nodeCount > 0, ≤ maxMLGraphNodeCount (VUID_ML_GRAPH_NODE_COUNT)
- DAG must be acyclic (VUID_ML_GRAPH_DAG)
- externalInputCount > 0 (VUID_ML_GRAPH_INPUT_COUNT)
- externalOutputCount > 0 (VUID_ML_GRAPH_OUTPUT_COUNT)
- Per-node descriptors must be valid for their operation type
- mlGraph feature must be enabled (VUID_ML_GRAPH_FEATURE)

**Deep copy requirements** (v1.0 fixes):
- `op_desc_size_by_stype()` must handle all 10 descriptor sTypes
- Reshape (`pOutputDimensions`) and transpose (`pPermutation`) need pointer deep-copy

### VkMLSessionKHR (Non-dispatchable handle)

Execution context for ML graph dispatches.

| Field | Type | Constraints |
|-------|------|-------------|
| graph | VkMLGraphKHR | Must be valid handle |
| scratchMemory | VkDeviceMemory | User-provided or auto-allocated |
| scratchMemoryOffset | VkDeviceSize | Aligned to minTensorMemoryAlignment |
| scratchMemorySize | VkDeviceSize | ≥ graph.scratchMemorySize |
| autoAllocated | VkBool32 | Tracks ownership for cleanup |

**Lifecycle**: Create (with graph + scratch) → Dispatch → Destroy (before graph)

**Validation rules**:
- graph must not be VK_NULL_HANDLE (VUID_SESSION_GRAPH_VALID)
- If scratch is explicit: size ≥ required (VUID_SESSION_SCRATCH_SIZE)
- If scratch is explicit: offset aligned (VUID_SESSION_SCRATCH_OFFSET_ALIGN)
- If scratch is VK_NULL_HANDLE: mlGraphScratchAutoAllocation required (VUID_SESSION_SCRATCH_AUTO)
- **v1.0 fix**: struct must be zero-initialized after allocation

## Primitive Descriptor Types

| sType Enum | Struct | Operation Types |
|-----------|--------|-----------------|
| ML_PRIMITIVE_DESC_CONVOLUTION_KHR (1000559014) | VkMLPrimitiveDescConvolutionKHR | CONVOLUTION_2D, DECONVOLUTION_2D |
| ML_PRIMITIVE_DESC_GEMM_KHR (1000559015) | VkMLPrimitiveDescGemmKHR | GEMM, FULLY_CONNECTED |
| ML_PRIMITIVE_DESC_POOLING_KHR (1000559016) | VkMLPrimitiveDescPoolingKHR | MAX_POOL_2D, AVERAGE_POOL_2D, GLOBAL_AVERAGE_POOL |
| ML_PRIMITIVE_DESC_ACTIVATION_KHR (1000559017) | VkMLPrimitiveDescActivationKHR | RELU, SIGMOID, TANH, LEAKY_RELU, PRELU, SOFTMAX |
| ML_PRIMITIVE_DESC_NORMALIZATION_KHR (1000559018) | VkMLPrimitiveDescNormalizationKHR | BATCH_NORMALIZATION, LRN |
| ML_PRIMITIVE_DESC_ELEMENTWISE_KHR (1000559019) | VkMLPrimitiveDescElementwiseKHR | ELEMENTWISE_ADD, ELEMENTWISE_MUL |
| ML_PRIMITIVE_DESC_CONCAT_KHR (1000559020) | VkMLPrimitiveDescConcatKHR | CONCAT |
| ML_PRIMITIVE_DESC_RESHAPE_KHR (1000559021) | VkMLPrimitiveDescReshapeKHR | RESHAPE |
| ML_PRIMITIVE_DESC_TRANSPOSE_KHR (1000559022) | VkMLPrimitiveDescTransposeKHR | TRANSPOSE |
| ML_PRIMITIVE_DESC_RESIZE_KHR (1000559023) | VkMLPrimitiveDescResizeKHR | RESIZE |

## VkStructureType Registry (v1.0 corrected)

| Value | Name | Status |
|-------|------|--------|
| 1000559000 | PHYSICAL_DEVICE_ML_FEATURES_KHR | Stable |
| 1000559001 | PHYSICAL_DEVICE_ML_PROPERTIES_KHR | Stable |
| 1000559002 | TENSOR_CREATE_INFO_KHR | Stable |
| 1000559003 | TENSOR_DESCRIPTION_KHR | Stable |
| 1000559004 | TENSOR_VIEW_CREATE_INFO_KHR | Stable |
| 1000559005 | TENSOR_MEMORY_BARRIER_KHR | Stable |
| 1000559006 | TENSOR_MEMORY_REQUIREMENTS_INFO_KHR | Stable |
| 1000559007 | BIND_TENSOR_MEMORY_INFO_KHR | Stable |
| 1000559008 | WRITE_DESCRIPTOR_SET_TENSOR_KHR | Stable |
| 1000559009 | TENSOR_FORMAT_PROPERTIES_KHR | Stable |
| 1000559010 | COPY_TENSOR_INFO_KHR | Stable |
| 1000559011 | ML_GRAPH_CREATE_INFO_KHR | Stable |
| 1000559012 | ML_GRAPH_NODE_CREATE_INFO_KHR | Stable |
| 1000559013 | ML_SESSION_CREATE_INFO_KHR | Stable |
| 1000559014 | ML_PRIMITIVE_DESC_CONVOLUTION_KHR | Stable |
| 1000559015 | ML_PRIMITIVE_DESC_GEMM_KHR | Stable |
| 1000559016 | ML_PRIMITIVE_DESC_POOLING_KHR | Stable |
| 1000559017 | ML_PRIMITIVE_DESC_ACTIVATION_KHR | Stable |
| 1000559018 | ML_PRIMITIVE_DESC_NORMALIZATION_KHR | Stable |
| 1000559019 | ML_PRIMITIVE_DESC_ELEMENTWISE_KHR | Stable |
| 1000559020 | ML_PRIMITIVE_DESC_CONCAT_KHR | Stable |
| 1000559021 | ML_PRIMITIVE_DESC_RESHAPE_KHR | Stable |
| 1000559022 | ML_PRIMITIVE_DESC_TRANSPOSE_KHR | Stable |
| 1000559023 | ML_PRIMITIVE_DESC_RESIZE_KHR | Stable |
| 1000559024 | ML_GRAPH_DISPATCH_INFO_KHR | **REASSIGNED** (was 1000559020) |
| 1000559025 | ML_TENSOR_BINDING_KHR | **REASSIGNED** (was 1000559021) |
| 1000559026 | TENSOR_DEPENDENCY_INFO_KHR | **REASSIGNED** (was 1000559022) |
| 1000559027 | TENSOR_COPY_KHR | **REASSIGNED** (was 1000559023) |

## State Transitions

### Tensor Lifecycle

```
CREATED ──bind memory──▶ BOUND ──use──▶ IN_USE ──idle──▶ BOUND ──destroy──▶ DESTROYED
                                                                    │
                                         ◀── views must be destroyed first ──┘
```

### Graph Lifecycle

```
CREATED ──query scratch──▶ QUERYABLE ──create sessions──▶ IN_USE ──destroy sessions──▶ QUERYABLE ──destroy──▶ DESTROYED
```

### Session Lifecycle

```
CREATED ──dispatch──▶ IN_USE ──idle──▶ CREATED ──destroy──▶ DESTROYED
```
