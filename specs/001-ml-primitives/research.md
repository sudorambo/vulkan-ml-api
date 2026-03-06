# Research: VK_KHR_ml_primitives

**Branch**: `001-ml-primitives` | **Date**: 2026-03-05

## Research Summary

No NEEDS CLARIFICATION markers exist — the authoritative specification
(`spec/VK_KHR_ml_primitives.adoc`) and project constitution fully define
all technical decisions. This document records the rationale behind each
key architectural choice and alternatives evaluated.

---

## R-001: Object Model — Why New Object Types vs. Pipeline Extensions

**Decision**: Introduce four new non-dispatchable handle types
(`VkTensorKHR`, `VkTensorViewKHR`, `VkMLGraphKHR`, `VkMLSessionKHR`)
rather than extending existing compute pipeline or buffer objects.

**Rationale**: Tensors have N-dimensional semantics (up to 8D) with
format, tiling, and ML-specific usage flags that do not map cleanly onto
`VkBuffer` (1D, untyped) or `VkImage` (limited to 3D + layers). ML
graphs require DAG compilation semantics absent from
`VkComputePipeline`. Sessions manage scratch memory and transient state
specific to graph execution.

**Alternatives considered**:

- *Repurpose VkBuffer with metadata*: Rejected — loses type safety and
  would require extensive `pNext` chains to carry dimension/format info
  on every API call, violating Principle VII (simplicity).
- *Extend VkImage to N-D*: Rejected — `VkImage` carries rendering
  semantics (mipmaps, multisampling, layouts) irrelevant to ML tensors,
  and 8D support would break existing image validation.

---

## R-002: Graph Compilation Model — Ahead-of-Time vs. JIT

**Decision**: ML graphs are compiled once at creation time
(`vkCreateMLGraphKHR`) and are immutable thereafter. Graph compilation
is a potentially expensive operation that may be deferred to background
threads via `VK_KHR_deferred_host_operations`.

**Rationale**: Ahead-of-time compilation enables IHV-specific
optimizations (operator fusion, memory planning, kernel selection)
without per-frame cost. Immutability simplifies thread safety — a
compiled graph can be shared across command buffers and sessions
without synchronization.

**Alternatives considered**:

- *JIT compilation at first dispatch*: Rejected — unpredictable first-
  frame latency violates the performance constraint (graph compilation
  is setup-time, not per-frame).
- *Mutable graphs*: Rejected — would require versioning, invalidation
  tracking, and re-compilation triggers, adding complexity without
  clear benefit (Principle VII).

---

## R-003: Synchronization — Reuse Existing Primitives vs. New Ones

**Decision**: Reuse `VkPipelineBarrier2` (with `VkTensorMemoryBarrierKHR`
chained via `VkTensorDependencyInfoKHR::pNext`) and timeline semaphores
for all synchronization. Introduce only a new pipeline stage
(`VK_PIPELINE_STAGE_2_ML_GRAPH_BIT_KHR`) and access flags.

**Rationale**: Constitution Principle VII mandates composability with
existing Vulkan primitives. A new pipeline stage is the minimum addition
needed to express ML-specific ordering. Everything else (barriers,
semaphores, queue submission) reuses existing infrastructure.

**Alternatives considered**:

- *Dedicated ML fence type*: Rejected — timeline semaphores already
  provide cross-queue signaling with monotonic values, covering all
  ML synchronization needs.
- *Implicit synchronization within sessions*: Rejected — violates
  Vulkan's explicit synchronization philosophy and Principle V
  (explicit lifecycle).

---

## R-004: Tensor Memory Model — Application-Managed vs. Driver-Managed

**Decision**: Application-managed memory as default (query requirements
→ allocate → bind), with opt-in driver-managed scratch via the
`mlGraphScratchAutoAllocation` feature flag.

**Rationale**: Application-managed memory is the Vulkan norm for images
and buffers, enabling suballocation, memory aliasing, and budget
control. The opt-in auto-allocation path serves simpler use cases
without violating the explicit-by-default model (Principle V).

**Alternatives considered**:

- *Fully driver-managed tensor memory*: Rejected — removes application
  control over memory budget and prevents suballocation strategies,
  violating Principle V.
- *No auto-allocation option*: Considered but rejected — scratch memory
  management is boilerplate that adds friction without value for simple
  inference scenarios.

---

## R-005: Validation Layer Architecture

**Decision**: Implement validation as a separate Vulkan layer
(`VK_LAYER_KHR_ml_validation`) that intercepts all ML extension entry
points, validates parameters against VUIDs, and forwards to the ICD.

**Rationale**: Standard Vulkan validation architecture. Separates
correctness checking from implementation, enabling validation in
debug builds without performance cost in release. Each VUID from the
spec maps to a discrete check function.

**Alternatives considered**:

- *Inline validation in ICD*: Rejected — couples validation with
  implementation, makes it impossible to disable for release builds,
  and violates the layered architecture pattern.

---

## R-006: Build System and Toolchain

**Decision**: CMake 3.20+ as the sole build system. Targets GCC 11+,
Clang 14+, MSVC 2022+. CI runs `clang-tidy` and `cppcheck` with zero
warnings enforced.

**Rationale**: CMake is the de facto standard for Vulkan ecosystem
projects (Vulkan-Loader, Vulkan-ValidationLayers, SPIRV-Tools all use
CMake). The compiler matrix covers all target platforms. Static analysis
tools are widely available and integrate with CMake via
`CMAKE_EXPORT_COMPILE_COMMANDS`.

**Alternatives considered**:

- *Meson*: Rejected — lower adoption in the Vulkan ecosystem; would
  create friction for contributors familiar with CMake.
- *Bazel*: Rejected — heavy dependency, uncommon in graphics/Vulkan
  projects.

---

## R-007: Test Strategy

**Decision**: Three-tier test structure: CTS (conformance), validation
(VUID negative tests), and unit (internal logic). CTS tests query
capabilities and skip gracefully. All tests run in CI.

**Rationale**: Constitution Principle IV mandates test-first with
validation layers. The three-tier structure maps to different failure
modes: CTS verifies correct behavior, validation tests verify error
detection, unit tests verify internal algorithms (DAG cycle detection,
shape inference).

**Alternatives considered**:

- *CTS-only testing*: Rejected — CTS tests correctness but not
  error handling. VUID negative tests are separately required by
  Principle IV.
- *Property-based testing*: Considered as a supplement for shape
  validation and tensor format compatibility. May be added in a
  future iteration but not required for initial implementation.

---

## R-008: SPIR-V Integration Approach

**Decision**: Co-maintain `SPV_KHR_tensor` extension alongside the
Vulkan extension. Tensor types and accessors (`OpTypeTensorKHR`,
`OpTensorReadKHR`, `OpTensorWriteKHR`, `OpTensorQuerySizeKHR`) are
defined in SPIR-V and mapped to GLSL via `GL_KHR_tensor`.

**Rationale**: SPIR-V integration enables hybrid workloads where custom
shaders and opaque ML primitives share tensor resources (FR-012). The
tensor type system in SPIR-V mirrors the Vulkan tensor description,
enabling zero-copy interop.

**Alternatives considered**:

- *Buffer-based shader access only*: Rejected — loses typed access and
  multi-dimensional indexing. Would require manual stride calculations
  in every shader.
- *Image-based aliasing only*: Rejected — limited to formats and
  dimensions supported by `VkImage`, missing 5D-8D tensors and ML-
  specific formats (BF16, FP8).
