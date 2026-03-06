# Data Model: VK_KHR_ml_primitives

**Branch**: `001-ml-primitives` | **Date**: 2026-03-05

## Entity Overview

```text
VkPhysicalDevice
 └─ queries ──► VkPhysicalDeviceMLFeaturesKHR
 └─ queries ──► VkPhysicalDeviceMLPropertiesKHR

VkDevice
 ├─ creates ──► VkTensorKHR ──► binds VkDeviceMemory
 │               └─ creates ──► VkTensorViewKHR
 ├─ creates ──► VkMLGraphKHR (from VkMLGraphNodeCreateInfoKHR[])
 │               └─ queries ──► VkMemoryRequirements2 (scratch)
 └─ creates ──► VkMLSessionKHR (binds VkMLGraphKHR + scratch memory)

VkCommandBuffer
 ├─ records ──► vkCmdDispatchMLGraphKHR(VkMLSessionKHR, tensors)
 ├─ records ──► vkCmdCopyTensorKHR(src, dst, regions)
 └─ records ──► vkCmdPipelineBarrier2(VkTensorMemoryBarrierKHR via pNext)
```

---

## Entities

### VkTensorKHR

N-dimensional data container for ML workloads.

| Field | Type | Constraints |
|-------|------|-------------|
| Handle | `VkTensorKHR` | Non-dispatchable. Unique per device. |
| Description | `VkTensorDescriptionKHR` | Embedded via pointer in create info. |
| Sharing Mode | `VkSharingMode` | EXCLUSIVE or CONCURRENT. |
| Queue Families | `uint32_t[]` | Required if CONCURRENT; each unique, < queue family count. |
| Bound Memory | `VkDeviceMemory` | Bound exactly once via `vkBindTensorMemoryKHR`. |

**Lifecycle**: Create → Query Memory Requirements → Allocate Memory →
Bind Memory → Use (dispatch/copy/shader) → Destroy.

**Validation rules** (VUIDs):
- `tensorObjects` feature MUST be enabled.
- Device MUST have at least one queue.
- Memory MUST NOT be bound twice.
- Memory offset MUST be aligned to `minTensorMemoryAlignment`.
- Memory type MUST be compatible with requirements.
- All views MUST be destroyed before the tensor.
- All submitted commands referencing the tensor MUST have completed.

---

### VkTensorDescriptionKHR

Describes geometry, format, layout, and usage of a tensor.

| Field | Type | Constraints |
|-------|------|-------------|
| tiling | `VkTensorTilingKHR` | OPTIMAL or LINEAR. |
| format | `VkFormat` | Single-component only. |
| dimensionCount | `uint32_t` | 1 to `maxTensorDimensions`. |
| pDimensions | `const uint32_t*` | Each > 0, each ≤ `maxTensorDimensionSize`. Product ≤ `maxTensorElements`. |
| pStrides | `const VkDeviceSize*` | NULL for dense packing. MUST be NULL when tiling is OPTIMAL. Each stride multiple of element size. |
| usage | `VkTensorUsageFlagsKHR` | Bitmask of: SHADER, TRANSFER_SRC, TRANSFER_DST, ML_GRAPH_INPUT, ML_GRAPH_OUTPUT, ML_GRAPH_WEIGHT, IMAGE_ALIASING. |

---

### VkTensorViewKHR

Typed sub-region view into a tensor.

| Field | Type | Constraints |
|-------|------|-------------|
| Handle | `VkTensorViewKHR` | Non-dispatchable. |
| tensor | `VkTensorKHR` | MUST be valid, MUST have memory bound. |
| format | `VkFormat` | Element size MUST equal tensor's format element size. |
| dimensionCount | `uint32_t` | MUST equal tensor's dimensionCount. |
| pDimensionOffsets | `const uint32_t*` | Per-dimension offset. offset + size ≤ tensor dimension. |
| pDimensionSizes | `const uint32_t*` | Per-dimension size of the view window. |

**Lifecycle**: Create (from bound tensor) → Use → Destroy (before parent tensor).

---

### VkMLGraphKHR

Compiled directed acyclic graph of ML primitive operations. Immutable
after creation.

| Field | Type | Constraints |
|-------|------|-------------|
| Handle | `VkMLGraphKHR` | Non-dispatchable. |
| nodeCount | `uint32_t` | > 0, ≤ `maxMLGraphNodeCount`. |
| pNodes | `VkMLGraphNodeCreateInfoKHR[]` | MUST form a valid DAG. |
| externalInputCount | `uint32_t` | > 0. |
| externalOutputCount | `uint32_t` | > 0. |
| constantWeightCount | `uint32_t` | ≥ 0. |
| External descriptions | `VkTensorDescriptionKHR[]` | Shape/format for each external tensor. |

**Lifecycle**: Create (compile) → Query Scratch Requirements → Create
Sessions → Dispatch (N times) → Destroy Sessions → Destroy Graph.

**Validation rules**:
- `mlGraph` feature MUST be enabled.
- Graph topology MUST be a DAG (no cycles).
- All internal tensor edges MUST have compatible shapes/formats.
- Node count MUST NOT exceed `maxMLGraphNodeCount`.
- All submitted commands and sessions MUST be complete/destroyed first.

---

### VkMLGraphNodeCreateInfoKHR

Defines a single node in the ML graph.

| Field | Type | Constraints |
|-------|------|-------------|
| operationType | `VkMLOperationTypeKHR` | One of 21 operation types. |
| pOperationDesc | `const void*` | Points to operation-specific descriptor. `sType` MUST match operation type. |
| inputCount | `uint32_t` | Operation-dependent. |
| pInputs | `VkMLTensorBindingKHR[]` | Tensor bindings for inputs (and weights). |
| outputCount | `uint32_t` | Operation-dependent. |
| pOutputs | `VkMLTensorBindingKHR[]` | Tensor bindings for outputs. |
| pNodeName | `const char*` | Optional debug label. May be NULL. |

---

### VkMLTensorBindingKHR

Describes how a tensor is connected within the graph.

| Field | Type | Constraints |
|-------|------|-------------|
| bindingType | `VkMLTensorBindingTypeKHR` | INTERNAL, EXTERNAL_INPUT, EXTERNAL_OUTPUT, or EXTERNAL_WEIGHT. |
| nodeIndex | `uint32_t` | Producer node index (for INTERNAL only). |
| tensorIndex | `uint32_t` | Output slot of producer (INTERNAL) or index into external description array (EXTERNAL_*). |
| pTensorDescription | `const VkTensorDescriptionKHR*` | MUST be consistent with referenced location. |

---

### VkMLSessionKHR

Execution context for ML graph dispatch.

| Field | Type | Constraints |
|-------|------|-------------|
| Handle | `VkMLSessionKHR` | Non-dispatchable. |
| graph | `VkMLGraphKHR` | MUST be valid. Created on same device. |
| scratchMemory | `VkDeviceMemory` | May be VK_NULL_HANDLE if `mlGraphScratchAutoAllocation` enabled. |
| scratchMemoryOffset | `VkDeviceSize` | Offset within scratch allocation. |
| scratchMemorySize | `VkDeviceSize` | MUST be ≥ graph scratch requirement. |

**Lifecycle**: Create (bind graph + scratch) → Dispatch (N times via
command buffers) → Destroy (after all dispatches complete).

---

### ML Primitive Descriptors

Six descriptor structure types, each with `sType`/`pNext`:

| Descriptor | Applies To | Key Parameters |
|------------|-----------|----------------|
| `VkMLPrimitiveDescConvolutionKHR` | CONVOLUTION_2D, DECONVOLUTION_2D | inputLayout, kernel size, stride, dilation, padding, groupCount, fusedActivation |
| `VkMLPrimitiveDescGemmKHR` | GEMM, FULLY_CONNECTED | transposeA/B, alpha, beta, fusedActivation |
| `VkMLPrimitiveDescPoolingKHR` | MAX_POOL_2D, AVERAGE_POOL_2D, GLOBAL_AVERAGE_POOL | poolType, window size, stride, padding |
| `VkMLPrimitiveDescActivationKHR` | RELU, SIGMOID, TANH, LEAKY_RELU, PRELU, SOFTMAX | activationType, param0, param1 |
| `VkMLPrimitiveDescNormalizationKHR` | BATCH_NORMALIZATION, LRN | normType, epsilon, fusedActivation |
| `VkMLPrimitiveDescElementwiseKHR` | ELEMENTWISE_ADD, ELEMENTWISE_MUL | opType, fusedActivation |

Operations without dedicated descriptors (CONCAT, RESHAPE, TRANSPOSE,
RESIZE) use `pOperationDesc = NULL` with shape information inferred
from tensor bindings.

---

### VkTensorMemoryBarrierKHR

Synchronization primitive for tensor access ordering.

| Field | Type | Constraints |
|-------|------|-------------|
| srcAccessMask | `VkAccessFlags2` | ML_GRAPH_READ/WRITE, SHADER_READ/WRITE, TRANSFER_READ/WRITE. |
| dstAccessMask | `VkAccessFlags2` | Same set. |
| srcQueueFamilyIndex | `uint32_t` | Queue family or VK_QUEUE_FAMILY_IGNORED. |
| dstQueueFamilyIndex | `uint32_t` | Queue family or VK_QUEUE_FAMILY_IGNORED. |
| tensor | `VkTensorKHR` | MUST be valid. |

Chained into `VkDependencyInfo::pNext` via `VkTensorDependencyInfoKHR`.

---

## Enumerations

| Enum | Values | Description |
|------|--------|-------------|
| `VkTensorTilingKHR` | OPTIMAL, LINEAR | Memory layout strategy. |
| `VkTensorUsageFlagBitsKHR` | SHADER, TRANSFER_SRC/DST, ML_GRAPH_INPUT/OUTPUT/WEIGHT, IMAGE_ALIASING | Tensor usage intent. |
| `VkMLOperationTypeKHR` | 21 values (0-20) | ML primitive type. |
| `VkMLActivationFunctionKHR` | NONE, RELU, SIGMOID, TANH, LEAKY_RELU, CLAMP | Fused activation type. |
| `VkMLPaddingModeKHR` | VALID, SAME, EXPLICIT | Padding strategy. |
| `VkMLTensorLayoutKHR` | NCHW, NHWC, NDHWC | Tensor dimension ordering. |
| `VkMLTensorBindingTypeKHR` | INTERNAL, EXTERNAL_INPUT, EXTERNAL_OUTPUT, EXTERNAL_WEIGHT | Graph edge type. |

## State Transitions

### Tensor State Machine

```text
CREATED ─── vkBindTensorMemoryKHR ───► BOUND ─── vkDestroyTensorKHR ───► DESTROYED
              (memory queryable)         (usable in dispatches,
                                          copies, shaders, views)
```

### ML Graph State Machine

```text
COMPILED ─── vkGetMLGraphMemoryRequirementsKHR ───► QUERYABLE ───► DISPATCHED (N times)
                                                                       │
                                                            vkDestroyMLGraphKHR
                                                                       │
                                                                       ▼
                                                                   DESTROYED
```

### ML Session State Machine

```text
CREATED ─── vkCmdDispatchMLGraphKHR (recorded) ───► IN_USE ───► IDLE
   │                                                              │
   │         (submit → GPU complete)                              │
   │                                                              │
   └──────── vkDestroyMLSessionKHR ◄──────────────────────────────┘
```
