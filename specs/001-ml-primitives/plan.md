# Implementation Plan: VK_KHR_ml_primitives

**Branch**: `001-ml-primitives` | **Date**: 2026-03-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-ml-primitives/spec.md`

## Summary

Implement the `VK_KHR_ml_primitives` Vulkan extension: a cross-vendor
ML execution framework introducing native tensor resources (`VkTensorKHR`),
21 IHV-optimized ML primitive operations, a compiled ML graph (`VkMLGraphKHR`)
abstraction for whole-model dispatch, and an ML session (`VkMLSessionKHR`)
for execution state management. The implementation consists of a public C
header defining the API surface, a reference ICD-layer implementation, a
validation layer enforcing all VUIDs from the specification, and a
conformance test suite. All code is C11, builds with CMake, and targets
Linux, Windows, and Android.

## Technical Context

**Language/Version**: C (C11 minimum; C17 preferred for implementation files)
**Primary Dependencies**: Vulkan 1.3, `VK_KHR_cooperative_matrix`,
`VK_KHR_timeline_semaphore` (core 1.2), `VK_KHR_maintenance5`,
`VK_KHR_format_feature_flags2`, `SPV_KHR_tensor`
**Storage**: N/A (GPU device memory managed via Vulkan memory allocator)
**Testing**: Vulkan CTS framework, validation layer test harness,
`clang-tidy`, `cppcheck`, GoogleTest (C wrapper for CTS harness)
**Target Platform**: Linux (x86_64, aarch64), Windows (x86_64),
Android (aarch64)
**Project Type**: Library (Vulkan extension: public header + ICD layer
implementation + validation layer + CTS)
**Performance Goals**: Zero host-memory allocation in dispatch
(`vkCmdDispatchMLGraphKHR`) and barrier hot paths; graph compilation
is a setup-time cost, not per-frame
**Constraints**: Thread safety per standard Vulkan external-sync model;
public headers compile cleanly under `-Wall -Wextra -Wpedantic` on
GCC 11+, Clang 14+, MSVC 2022+; no compiler-specific extensions in
public headers
**Scale/Scope**: ~15 API entry points, 4 new object types, 7 new
enumerations, 21 ML operation types, 6 primitive descriptor structures,
~50 VUIDs, 25 `VkStructureType` values

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| I | Specification-Driven Development | PASS | The `.adoc` spec is the sole source of truth. All implementation artifacts (header, validation, CTS) trace to spec sections and VUIDs. |
| II | Vulkan C API Conventions | PASS | C11 language, `VK_`/`Vk`/`vk` naming, `sType`/`pNext` structure patterns, `VkResult` returns, `VkAllocationCallbacks*` on all creators, explicit create/destroy pairs — all defined in the spec and enforced in this plan. |
| III | Portability and Cross-Vendor | PASS | All optional features queryable via `VkPhysicalDeviceMLFeaturesKHR`/`VkPhysicalDeviceMLPropertiesKHR`. No vendor assumptions. Platform-independent types throughout. |
| IV | Test-First with Validation | PASS | Plan mandates: VUID validation checks before implementation is considered complete; CTS covers every entry point, enum value, and error path; static analysis with zero warnings. |
| V | Explicit Resource Lifecycle | PASS | Tensor/graph/session follow image/buffer lifecycle. Memory requirements queryable before allocation. Scratch auto-alloc is opt-in via capability flag. |
| VI | Backward Compatibility | PASS | Revision 1 (initial). All `VkStructureType` values registered. Extension mechanism (`pNext`) used for future evolution. |
| VII | Simplicity and Composability | PASS | Reuses `VkCommandBuffer`, `vkQueueSubmit2`, `VkPipelineBarrier2`, timeline semaphores. No new synchronization primitives. Tensor lifecycle mirrors image/buffer. |

**Gate result**: ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-ml-primitives/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (public C API contract)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
include/
└── vulkan/
    └── vulkan_ml_primitives.h       # Public C header: types, enums, prototypes

src/
├── internal.h                       # Internal shared declarations
├── tensor.c                         # VkTensorKHR create/destroy/memory
├── tensor_view.c                    # VkTensorViewKHR create/destroy
├── tensor_copy.c                    # vkCmdCopyTensorKHR
├── tensor_barrier.c                 # VkTensorMemoryBarrierKHR handling
├── ml_primitives.c                  # Primitive descriptor validation/setup
├── ml_graph.c                       # VkMLGraphKHR create/destroy/memory query
├── ml_session.c                     # VkMLSessionKHR create/destroy
├── ml_dispatch.c                    # vkCmdDispatchMLGraphKHR
└── feature_query.c                  # Feature/property query entry points

layers/
└── validation/
    ├── vk_ml_validation.h           # Validation layer shared header
    ├── tensor_validation.c          # VUID checks for tensor operations
    ├── graph_validation.c           # VUID checks for graph operations
    ├── session_validation.c         # VUID checks for session operations
    └── dispatch_validation.c        # VUID checks for dispatch operations

tests/
├── cts/
│   ├── test_tensor_lifecycle.c      # Tensor create/bind/destroy
│   ├── test_tensor_view.c           # Tensor view creation and access
│   ├── test_tensor_copy.c           # Tensor copy operations
│   ├── test_tensor_formats.c        # Format support and feature queries
│   ├── test_ml_graph.c              # Graph construction and compilation
│   ├── test_ml_session.c            # Session create/destroy
│   ├── test_ml_dispatch.c           # Graph dispatch execution
│   ├── test_synchronization.c       # Barriers and semaphore interop
│   └── test_spirv_tensor.c          # SPIR-V tensor shader access
├── validation/
│   └── test_vuids.c                 # Negative tests for all VUIDs
└── unit/
    ├── test_dag_validation.c        # DAG cycle detection and shape checks
    └── test_descriptor_validation.c # Primitive descriptor parameter checks

spec/
└── VK_KHR_ml_primitives.adoc       # Authoritative specification (exists)

CMakeLists.txt                       # Root build file
```

**Structure Decision**: Single-project layout appropriate for a Vulkan
extension library. The `include/` directory contains the public header
installed by consumers. The `src/` directory contains the reference ICD
layer implementation. The `layers/validation/` directory contains the
validation layer (loadable separately). The `tests/` directory is
organized by test type: conformance (CTS), validation (VUID negative
tests), and unit (internal logic tests).

## Complexity Tracking

No constitution violations. No complexity justifications needed.
