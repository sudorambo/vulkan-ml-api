# Public API Contract: VK_KHR_ml_primitives

**Version**: 1.0.0 (pre-release, subject to freeze at 1.0 tag)
**Header**: `include/vulkan/vulkan_ml_primitives.h`

This document defines the public contract of the `VK_KHR_ml_primitives`
extension as implemented by the reference ICD and validated by the
validation layer. All contracts are derived from the extension specification
and must remain stable after 1.0 release.

## 1. API Entry Points

### 1.1 Tensor Lifecycle

| Function | Returns | Preconditions | Postconditions |
|----------|---------|---------------|----------------|
| `vkCreateTensorKHR` | `VkResult` | `tensorObjects` feature enabled; valid `pCreateInfo` with `sType == TENSOR_CREATE_INFO_KHR` | Tensor handle allocated; no memory bound |
| `vkDestroyTensorKHR` | void | Tensor not in use by any submitted command; all views destroyed | Handle invalidated; host memory freed |
| `vkCreateTensorViewKHR` | `VkResult` | Parent tensor has memory bound; valid `pCreateInfo` with `sType == TENSOR_VIEW_CREATE_INFO_KHR` | View handle allocated |
| `vkDestroyTensorViewKHR` | void | View not in use by any submitted command | Handle invalidated; host memory freed |

### 1.2 Tensor Memory

| Function | Returns | Preconditions | Postconditions |
|----------|---------|---------------|----------------|
| `vkGetTensorMemoryRequirementsKHR` | void | Valid tensor handle | `pMemoryRequirements` populated |
| `vkBindTensorMemoryKHR` | `VkResult` | Each tensor not already bound; memory offset aligned | Tensors have memory bound; `memoryBound == VK_TRUE` |

### 1.3 Tensor Commands

| Function | Returns | Preconditions | Postconditions |
|----------|---------|---------------|----------------|
| `vkCmdCopyTensorKHR` | void | Command buffer in recording state; src has TRANSFER_SRC; dst has TRANSFER_DST; both have memory bound | Copy command recorded |

### 1.4 ML Graph

| Function | Returns | Preconditions | Postconditions |
|----------|---------|---------------|----------------|
| `vkCreateMLGraphKHR` | `VkResult` | `mlGraph` feature enabled; valid DAG; all primitive descriptors valid | Graph handle allocated; scratchMemorySize computed |
| `vkDestroyMLGraphKHR` | void | All sessions destroyed; graph not in use | Handle invalidated; all deep-copied data freed |
| `vkGetMLGraphMemoryRequirementsKHR` | void | Valid graph handle | Scratch memory requirements populated |

### 1.5 ML Session

| Function | Returns | Preconditions | Postconditions |
|----------|---------|---------------|----------------|
| `vkCreateMLSessionKHR` | `VkResult` | Valid graph; scratch ≥ required OR auto-alloc enabled | Session handle allocated |
| `vkDestroyMLSessionKHR` | void | Session not in use by any submitted command | Handle invalidated; auto-allocated scratch freed |

### 1.6 ML Dispatch

| Function | Returns | Preconditions | Postconditions |
|----------|---------|---------------|----------------|
| `vkCmdDispatchMLGraphKHR` | void | Command buffer recording; valid session; tensor counts match graph | Dispatch command recorded |

## 2. Error Contracts

All `VkResult`-returning functions follow these error precedence rules:

1. `VK_ERROR_UNKNOWN` — Invalid parameters (NULL pointers, wrong sType, invalid handles)
2. `VK_ERROR_OUT_OF_HOST_MEMORY` — Host allocation failure
3. `VK_ERROR_OUT_OF_DEVICE_MEMORY` — Device allocation failure (not applicable in reference ICD)
4. `VK_ERROR_INITIALIZATION_FAILED` — Internal initialization failure

Void-returning functions silently return on NULL/invalid input. They MUST NOT crash.

## 3. Handle Contracts

- `VK_NULL_HANDLE` is valid input to all destroy functions (no-op).
- `VK_NULL_HANDLE` is **invalid** input to all other functions and must return error or validation failure.
- Handles are non-dispatchable (`uint64_t` on 64-bit, `struct*` on 32-bit).
- Handle validity is not checked by the ICD beyond NULL (validation layer responsibility).

## 4. sType Contract

Every extensible structure's `sType` field must match exactly one value from the
registry. The ICD and validation layer both verify sType for all `pCreateInfo`
parameters.

**Post-1.0 invariant**: sType numeric values MUST NOT change. The corrected
registry (see [data-model.md](../data-model.md)) assigns values 1000559000–1000559027.

## 5. Memory Ownership Contract

- All `pCreateInfo` structures and their referenced memory may be freed by the
  caller immediately after `vkCreate*` returns. The ICD deep-copies all
  pointer members.
- `VkAllocationCallbacks` provided at creation must be provided (or NULL) at
  destruction. Mismatched callbacks produce undefined behavior.
- `vk_ml_alloc` returns uninitialized memory. Callers must initialize all fields
  before use. (v1.0 fix: session struct is now zero-initialized.)

## 6. Validation Layer Contract

The validation layer (`vk_ml_validation`) exposes functions that return
`VkBool32`: `VK_TRUE` for valid input, `VK_FALSE` for invalid.

| Function | Validates |
|----------|-----------|
| `vk_ml_validate_tensor_create` | Tensor creation parameters + feature/property limits |
| `vk_ml_validate_tensor_view_create` | View parameters + parent tensor state |
| `vk_ml_validate_tensor_bind` | Bind parameters + alignment + double-bind prevention |
| `vk_ml_validate_tensor_copy` | Copy info sType, handles, regions, offsets, extents |
| `vk_ml_validate_graph_create` | Graph DAG structure, node count, per-node descriptors |
| `vk_ml_validate_convolution_desc` | Convolution parameters (kernel, stride, dilation, padding, groups) |
| `vk_ml_validate_gemm_desc` | GEMM parameters (alpha, beta finiteness, fused activation) |
| `vk_ml_validate_pooling_desc` | Pooling parameters (type, window, stride) |
| `vk_ml_validate_activation_desc` | **NEW in 1.0**: Activation type range, param finiteness, clamp ordering |
| `vk_ml_validate_normalization_desc` | Normalization parameters (epsilon, type, fused activation) |
| `vk_ml_validate_elementwise_desc` | Elementwise parameters (op type, fused activation) |
| `vk_ml_validate_session_create` | Session parameters (graph handle, scratch size/alignment, auto-alloc) |
| `vk_ml_validate_dispatch` | Dispatch parameters (session, tensor counts, tensor arrays) |
| `vk_ml_validate_tensor_memory_barrier` | Barrier sType and access masks |
| `vk_ml_validate_tensor_dependency_info` | Dependency info sType and barrier array |

## 7. CMake Integration Contract

```cmake
find_package(vk_ml_primitives 1.0 REQUIRED)
target_link_libraries(app PRIVATE vk_ml_primitives::vk_ml_primitives)
target_link_libraries(app PRIVATE vk_ml_primitives::vk_ml_validation)  # optional
```

**Post-1.0 invariant**: Target names `vk_ml_primitives::vk_ml_primitives` and
`vk_ml_primitives::vk_ml_validation` are stable.

## 8. pkg-config Contract

```
pkg-config --cflags --libs vk_ml_primitives
```

Returns correct `-I` and `-L` flags for the install prefix, including
`lib64` on systems that use it.
