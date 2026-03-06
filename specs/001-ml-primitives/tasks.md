# Tasks: VK_KHR_ml_primitives

**Input**: Design documents from `/specs/001-ml-primitives/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/c-api.md, quickstart.md

**Tests**: Included — Constitution Principle IV (Test-First with Validation Layers) is NON-NEGOTIABLE.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Public header**: `include/vulkan/vulkan_ml_primitives.h`
- **Implementation**: `src/*.c`, `src/internal.h`
- **Validation layer**: `layers/validation/*.c`, `layers/validation/vk_ml_validation.h`
- **CTS tests**: `tests/cts/test_*.c`
- **Validation tests**: `tests/validation/test_vuids.c`
- **Unit tests**: `tests/unit/test_*.c`

---

## Phase 1: Setup

**Purpose**: Project initialization, build system, and directory structure

- [x] T001 Create project directory structure per plan.md (include/vulkan/, src/, layers/validation/, tests/cts/, tests/validation/, tests/unit/)
- [x] T002 Create root CMakeLists.txt with C11 standard, Vulkan 1.3 dependency, compiler warning flags (-Wall -Wextra -Wpedantic), and targets for library, validation layer, and test executables
- [x] T003 [P] Configure clang-tidy and cppcheck integration in CMakeLists.txt with zero-warning enforcement

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Public API header and shared infrastructure that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Define all VkStructureType extension values (25 values) and VkObjectType extensions in include/vulkan/vulkan_ml_primitives.h
- [x] T005 [P] Define all enumerations (VkTensorTilingKHR, VkTensorUsageFlagBitsKHR, VkMLOperationTypeKHR, VkMLActivationFunctionKHR, VkMLPaddingModeKHR, VkMLTensorLayoutKHR, VkMLTensorBindingTypeKHR) and bitmask typedefs in include/vulkan/vulkan_ml_primitives.h
- [x] T006 [P] Define all handle types (VkTensorKHR, VkTensorViewKHR, VkMLGraphKHR, VkMLSessionKHR) and extension constants (VK_KHR_ML_PRIMITIVES_SPEC_VERSION, VK_KHR_ML_PRIMITIVES_EXTENSION_NAME) in include/vulkan/vulkan_ml_primitives.h
- [x] T007 Define all structure types (VkTensorDescriptionKHR, VkTensorCreateInfoKHR, VkTensorViewCreateInfoKHR, VkTensorMemoryRequirementsInfoKHR, VkBindTensorMemoryInfoKHR, VkCopyTensorInfoKHR, VkTensorCopyKHR, VkTensorMemoryBarrierKHR, VkTensorDependencyInfoKHR, VkWriteDescriptorSetTensorKHR, VkTensorFormatPropertiesKHR, all 6 ML primitive descriptors, VkMLTensorBindingKHR, VkMLGraphNodeCreateInfoKHR, VkMLGraphCreateInfoKHR, VkMLSessionCreateInfoKHR, VkMLGraphDispatchInfoKHR, VkPhysicalDeviceMLFeaturesKHR, VkPhysicalDeviceMLPropertiesKHR) in include/vulkan/vulkan_ml_primitives.h
- [x] T008 Define all function prototypes (15 entry points per contracts/c-api.md) in include/vulkan/vulkan_ml_primitives.h
- [x] T009 Create internal shared declarations (internal object structs, helper macros, VUID string constants) in src/internal.h
- [x] T010 Implement VkPhysicalDeviceMLFeaturesKHR query in src/feature_query.c (FR-013)
- [x] T011 [P] Implement VkPhysicalDeviceMLPropertiesKHR query with all numeric limits in src/feature_query.c (FR-014)
- [x] T012 [P] Create validation layer shared header with intercept function declarations and VUID string table in layers/validation/vk_ml_validation.h

**Checkpoint**: Public header compiles cleanly on GCC/Clang/MSVC. Feature queries return valid data. Validation layer skeleton exists.

---

## Phase 3: User Story 1 — Tensor Resource Management (Priority: P1) MVP

**Goal**: Create, configure, bind memory to, copy, and destroy N-dimensional tensor resources.

**Independent Test**: Create tensors of various shapes (1D-8D) and formats (FP16, BF16, INT8, FP8), bind memory, verify memory requirements, create views, copy between tensors, destroy without errors.

### Tests for User Story 1

> **Write these tests FIRST — ensure they FAIL before implementation**

- [x] T013 [P] [US1] CTS test for tensor create/bind/destroy lifecycle with multiple formats and dimensions in tests/cts/test_tensor_lifecycle.c
- [x] T014 [P] [US1] CTS test for tensor view create/destroy with sub-region access and format reinterpretation in tests/cts/test_tensor_view.c
- [x] T015 [P] [US1] CTS test for tensor copy operations (single-region, multi-region, dimension validation) in tests/cts/test_tensor_copy.c
- [x] T016 [P] [US1] CTS test for tensor format property queries and feature-dependent format support in tests/cts/test_tensor_formats.c
- [x] T017 [P] [US1] Validation layer VUID tests for tensor operations (zero dimensions, exceeded limits, double bind, unbound access, invalid format) in tests/validation/test_vuids.c (tensor section)

### Implementation for User Story 1

- [x] T018 [US1] Implement vkCreateTensorKHR and vkDestroyTensorKHR in src/tensor.c (FR-001)
- [x] T019 [US1] Implement vkGetTensorMemoryRequirementsKHR and vkBindTensorMemoryKHR in src/tensor.c (FR-002)
- [x] T020 [P] [US1] Implement vkCreateTensorViewKHR and vkDestroyTensorViewKHR in src/tensor_view.c (FR-003)
- [x] T021 [P] [US1] Implement vkCmdCopyTensorKHR with region validation in src/tensor_copy.c (FR-004)
- [x] T022 [P] [US1] Implement VkTensorFormatPropertiesKHR query for tensor format capabilities in src/feature_query.c (FR-015)
- [x] T023 [US1] Implement tensor VUID validation checks (VUID-vkCreateTensorKHR-*, VUID-VkTensorDescriptionKHR-*, VUID-VkBindTensorMemoryInfoKHR-*, VUID-vkCmdCopyTensorKHR-*, VUID-VkTensorViewCreateInfoKHR-*) in layers/validation/tensor_validation.c

**Checkpoint**: Tensors can be created in all supported formats and dimensions, memory bound, views created, data copied. All tensor VUIDs enforced by validation layer. CTS tensor tests pass.

---

## Phase 4: User Story 2 — ML Graph Construction and Compilation (Priority: P2)

**Goal**: Define ML primitive descriptors, compose them into a DAG, compile the graph, and query scratch memory requirements.

**Independent Test**: Construct single-node and multi-node graphs with all 21 operation types, verify compilation succeeds for valid graphs, fails for invalid graphs (cycles, shape mismatches), and scratch requirements are queryable.

### Tests for User Story 2

> **Write these tests FIRST — ensure they FAIL before implementation**

- [x] T024 [P] [US2] CTS test for single-node graphs with each of the 21 ML operation types in tests/cts/test_ml_graph.c
- [x] T025 [P] [US2] CTS test for multi-node graph compilation (conv → batchnorm → relu → pool chain) with internal tensor edges in tests/cts/test_ml_graph.c
- [x] T026 [P] [US2] CTS test for scratch memory requirement queries and graph reuse across sessions in tests/cts/test_ml_graph.c
- [x] T027 [P] [US2] Unit test for DAG cycle detection algorithm and shape inference validation in tests/unit/test_dag_validation.c
- [x] T028 [P] [US2] Unit test for all 6 primitive descriptor parameter validation (stride > 0, dilation > 0, finite floats, valid enum values) in tests/unit/test_descriptor_validation.c
- [x] T029 [P] [US2] Validation layer VUID tests for graph operations (invalid DAG, exceeded node count, shape mismatch, invalid operation desc sType) in tests/validation/test_vuids.c (graph section)

### Implementation for User Story 2

- [x] T030 [US2] Implement convolution and GEMM primitive descriptor validation and setup in src/ml_primitives.c (FR-005, FR-007)
- [x] T031 [P] [US2] Implement pooling, activation, normalization, and elementwise primitive descriptor validation and setup in src/ml_primitives.c (FR-005)
- [x] T032 [US2] Implement DAG topology validation (cycle detection, depth check, edge compatibility) in src/ml_graph.c
- [x] T033 [US2] Implement vkCreateMLGraphKHR (graph compilation with node array, external tensor descriptions, weight descriptions) in src/ml_graph.c (FR-006)
- [x] T034 [US2] Implement vkDestroyMLGraphKHR in src/ml_graph.c
- [x] T035 [US2] Implement vkGetMLGraphMemoryRequirementsKHR for scratch memory queries in src/ml_graph.c
- [x] T036 [US2] Implement graph VUID validation checks (VUID-VkMLGraphCreateInfoKHR-*, VUID-VkMLPrimitiveDescConvolutionKHR-*, VUID-VkMLPrimitiveDescGemmKHR-*, VUID-VkMLPrimitiveDescPoolingKHR-*, VUID-VkMLPrimitiveDescNormalizationKHR-*, VUID-VkMLPrimitiveDescElementwiseKHR-*) in layers/validation/graph_validation.c

**Checkpoint**: All 21 operation types can be constructed into graphs. Multi-node DAGs compile with shape inference. Invalid graphs rejected. Scratch requirements queryable. All graph VUIDs enforced.

---

## Phase 5: User Story 3 — ML Session and Graph Dispatch (Priority: P3)

**Goal**: Create execution sessions, bind tensors at dispatch time, record dispatch commands into command buffers, and submit to compute queues.

**Independent Test**: Create a session for a single-convolution graph, dispatch with real tensors, submit to queue, verify output tensor is written. Test auto-allocation path and concurrent sessions.

### Tests for User Story 3

> **Write these tests FIRST — ensure they FAIL before implementation**

- [x] T037 [P] [US3] CTS test for session create/destroy with explicit scratch memory in tests/cts/test_ml_session.c
- [x] T038 [P] [US3] CTS test for session with auto-allocation (mlGraphScratchAutoAllocation feature query and fallback) in tests/cts/test_ml_session.c
- [x] T039 [P] [US3] CTS test for vkCmdDispatchMLGraphKHR with single-node convolution graph end-to-end (create tensors → build graph → session → dispatch → verify output) in tests/cts/test_ml_dispatch.c
- [x] T040 [P] [US3] CTS test for concurrent dispatch of multiple sessions for the same graph in tests/cts/test_ml_dispatch.c
- [x] T041 [P] [US3] Validation layer VUID tests for session and dispatch operations (insufficient scratch, mismatched tensor counts, wrong usage flags, dispatch inside render pass) in tests/validation/test_vuids.c (session/dispatch section)

### Implementation for User Story 3

- [x] T042 [US3] Implement vkCreateMLSessionKHR with scratch memory binding and auto-allocation support in src/ml_session.c (FR-008)
- [x] T043 [US3] Implement vkDestroyMLSessionKHR in src/ml_session.c
- [x] T044 [US3] Implement vkCmdDispatchMLGraphKHR command recording with tensor count and usage validation in src/ml_dispatch.c (FR-009)
- [x] T045 [US3] Implement session VUID validation checks (VUID-VkMLSessionCreateInfoKHR-*, VUID-vkCmdDispatchMLGraphKHR-*) in layers/validation/session_validation.c
- [x] T046 [US3] Implement dispatch VUID validation checks (tensor count matching, usage flag checks, render pass exclusion) in layers/validation/dispatch_validation.c

**Checkpoint**: Full end-to-end ML inference pipeline works: create tensors → compile graph → create session → dispatch → GPU produces output. Auto-allocation path functional. Concurrent sessions work independently. All session/dispatch VUIDs enforced.

---

## Phase 6: User Story 4 — Synchronization and Interop (Priority: P4)

**Goal**: Insert tensor memory barriers between ML dispatches and other Vulkan pipeline stages. Support cross-queue synchronization via timeline semaphores and queue family ownership transfers.

**Independent Test**: Dispatch ML graph, insert VkTensorMemoryBarrierKHR via VkTensorDependencyInfoKHR, then read output from compute shader. Verify barrier prevents data races. Test cross-queue with timeline semaphore.

### Tests for User Story 4

> **Write these tests FIRST — ensure they FAIL before implementation**

- [x] T047 [P] [US4] CTS test for tensor memory barrier (ML write → shader read transition) within single command buffer in tests/cts/test_synchronization.c
- [x] T048 [P] [US4] CTS test for cross-queue ML→compute synchronization using timeline semaphores in tests/cts/test_synchronization.c
- [x] T049 [P] [US4] CTS test for tensor queue family ownership transfer via paired VkTensorMemoryBarrierKHR in tests/cts/test_synchronization.c

### Implementation for User Story 4

- [x] T050 [US4] Implement VkTensorMemoryBarrierKHR and VkTensorDependencyInfoKHR processing in vkCmdPipelineBarrier2 interception in src/tensor_barrier.c (FR-010, FR-011)
- [x] T051 [US4] Implement VK_PIPELINE_STAGE_2_ML_GRAPH_BIT_KHR and VK_ACCESS_2_ML_GRAPH_READ/WRITE_BIT_KHR support in pipeline stage validation in src/tensor_barrier.c
- [x] T052 [US4] Implement tensor queue family ownership transfer validation (srcQueueFamilyIndex/dstQueueFamilyIndex consistency) in src/tensor_barrier.c

**Checkpoint**: Tensor memory barriers correctly order ML dispatch → shader/transfer reads. Cross-queue synchronization via timeline semaphores works. Queue family ownership transfers validated.

---

## Phase 7: User Story 5 — SPIR-V Tensor Shader Access (Priority: P5)

**Goal**: Enable compute shaders to access tensor data via SPIR-V tensor accessors (OpTensorReadKHR, OpTensorWriteKHR, OpTensorQuerySizeKHR) using the VK_DESCRIPTOR_TYPE_TENSOR_KHR descriptor type.

**Independent Test**: Create a tensor with SHADER usage, bind as descriptor, dispatch a compute shader that reads/writes tensor elements via SPIR-V, verify correct access. Test dynamic dimension queries.

### Tests for User Story 5

> **Write these tests FIRST — ensure they FAIL before implementation**

- [x] T053 [P] [US5] CTS test for VK_DESCRIPTOR_TYPE_TENSOR_KHR descriptor write and binding with VkWriteDescriptorSetTensorKHR in tests/cts/test_spirv_tensor.c
- [x] T054 [P] [US5] CTS test for compute shader reading/writing tensor elements via SPIR-V OpTensorReadKHR/OpTensorWriteKHR in tests/cts/test_spirv_tensor.c
- [x] T055 [P] [US5] CTS test for OpTensorQuerySizeKHR returning correct dimension sizes at runtime in tests/cts/test_spirv_tensor.c

### Implementation for User Story 5

- [x] T056 [US5] Implement VK_DESCRIPTOR_TYPE_TENSOR_KHR handling and VkWriteDescriptorSetTensorKHR processing in descriptor set updates in src/tensor.c (FR-012)
- [x] T057 [US5] Implement VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_KHR and VK_FORMAT_FEATURE_2_TENSOR_IMAGE_ALIASING_BIT_KHR format feature reporting in src/feature_query.c
- [x] T058 [US5] Implement tensorShaderAccess and tensorImageAliasing feature gate validation in layers/validation/tensor_validation.c

**Checkpoint**: Compute shaders can read and write tensor elements via SPIR-V. Descriptor binding works. Dynamic dimension queries return correct sizes. Feature gates enforced.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Comprehensive VUID coverage, documentation, static analysis, and integration validation

- [x] T059 [P] Complete VUID negative test coverage for ALL remaining VUIDs not covered in story-specific validation tests in tests/validation/test_vuids.c
- [x] T060 [P] Add Doxygen-style documentation comments to all public functions, structures, and enumerations in include/vulkan/vulkan_ml_primitives.h (mirroring spec language)
- [x] T061 [P] Run clang-tidy and cppcheck on all source files; fix any warnings to achieve zero-warning CI
- [x] T062 Run quickstart.md validation: execute the minimal convolution example end-to-end and verify it matches documented behavior
- [x] T063 Verify public header compiles cleanly with -Wall -Wextra -Wpedantic on GCC 11, Clang 14, and MSVC 2022

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational — BLOCKS US2, US3, US4, US5 (tensors are required by all)
- **User Story 2 (Phase 4)**: Depends on US1 (graphs reference tensor descriptions)
- **User Story 3 (Phase 5)**: Depends on US1 + US2 (sessions bind graphs and tensors)
- **User Story 4 (Phase 6)**: Depends on US1 + US3 (barriers synchronize dispatches)
- **User Story 5 (Phase 7)**: Depends on US1 (shader access operates on tensors)
- **Polish (Phase 8)**: Depends on all user stories

### User Story Dependencies

```text
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational: header + feature queries + validation skeleton)
    │
    ▼
Phase 3 (US1: Tensors) ◄── MVP STOP POINT
    │
    ├──────────────────────────┐
    ▼                          ▼
Phase 4 (US2: Graphs)    Phase 7 (US5: SPIR-V)
    │                     [independent of US2-US4]
    ▼
Phase 5 (US3: Sessions + Dispatch)
    │
    ▼
Phase 6 (US4: Synchronization)
    │
    ▼
Phase 8 (Polish)
```

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Validation layer checks are part of implementation (not separate phase)
- Models/internal structures before public API functions
- Create functions before destroy functions
- Query functions before bind/use functions

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (T005, T006, T011, T012)
- All CTS test files within a story can be written in parallel (different files)
- US2 and US5 can proceed in parallel after US1 completes
- Within US2: convolution/GEMM descriptors (T030) parallel with pooling/activation/norm/elementwise (T031)

---

## Parallel Example: User Story 1

```bash
# Write all CTS tests in parallel (different files):
Task: "CTS test for tensor lifecycle in tests/cts/test_tensor_lifecycle.c"
Task: "CTS test for tensor views in tests/cts/test_tensor_view.c"
Task: "CTS test for tensor copy in tests/cts/test_tensor_copy.c"
Task: "CTS test for tensor formats in tests/cts/test_tensor_formats.c"
Task: "VUID tests for tensor operations in tests/validation/test_vuids.c"

# Then implement in parallel where possible:
Task: "Implement tensor view in src/tensor_view.c"        # [P] - different file
Task: "Implement tensor copy in src/tensor_copy.c"        # [P] - different file
Task: "Implement tensor format query in src/feature_query.c"  # [P] - different file
```

---

## Parallel Example: After US1, US2 and US5 in parallel

```bash
# US2 and US5 have no dependency on each other:
# Developer A works on US2 (graphs):
Task: "CTS tests for ML graph in tests/cts/test_ml_graph.c"
Task: "Implement primitive descriptors in src/ml_primitives.c"
Task: "Implement graph compilation in src/ml_graph.c"

# Developer B works on US5 (SPIR-V) simultaneously:
Task: "CTS tests for SPIR-V tensor access in tests/cts/test_spirv_tensor.c"
Task: "Implement tensor descriptor type in src/tensor.c"
Task: "Implement format feature reporting in src/feature_query.c"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (header, feature queries)
3. Complete Phase 3: User Story 1 (Tensors)
4. **STOP and VALIDATE**: Create tensors, bind memory, copy data, create views
5. All tensor VUIDs enforced, CTS tests pass

### Incremental Delivery

1. Setup + Foundational → Header compiles, feature queries work
2. US1 (Tensors) → Tensor lifecycle complete, independently testable (MVP!)
3. US2 (Graphs) → Graph compilation works, all 21 ops validated
4. US3 (Sessions + Dispatch) → Full end-to-end ML inference pipeline
5. US4 (Synchronization) → Hybrid rendering/inference workloads
6. US5 (SPIR-V) → Custom shader access to tensor data
7. Polish → Full VUID coverage, documentation, static analysis

### Parallel Team Strategy

With multiple developers after US1:

1. Team completes Setup + Foundational + US1 together
2. Once US1 is done:
   - Developer A: US2 (Graphs) → US3 (Sessions)
   - Developer B: US5 (SPIR-V, independent of US2-US4)
3. After US3 complete:
   - Developer A: US4 (Synchronization)
4. Polish phase: all developers

---

## Phase 9: Remediation (Analysis Issues C1-C4)

**Purpose**: Fix 4 HIGH-severity gaps identified by `/speckit.analyze` where tasks were marked complete but underlying code is stubbed.

**Prerequisites**: All previous phases complete. No mutual dependencies between C1-C4 — all can run in parallel.

### C1: DAG Cycle Detection (FR-006, VUID_ML_GRAPH_DAG)

**Goal**: Replace DAG validation stub with DFS-based cycle detection. Reject cyclic graphs and invalid internal tensor edge references.

> **Write tests FIRST — ensure they FAIL before implementation**

- [x] T064 [P] Add test_cyclic_graph (A→B→A back edge), test_self_reference (node inputs reference itself), and test_invalid_node_index (nodeIndex >= nodeCount) to tests/unit/test_dag_validation.c
- [x] T065 [P] Add test_graph_cyclic_vuid negative test (valid graph structure but cyclic dependency → VK_FALSE) to tests/validation/test_vuids.c
- [x] T066 Implement DFS three-color cycle detection in vk_ml_validate_graph_create (replace stub at lines 36-37) in layers/validation/graph_validation.c — use stack-allocated uint8_t color[VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT], validate all INTERNAL binding nodeIndex < nodeCount, detect back edges (GRAY→GRAY)

**Checkpoint**: Cyclic graphs rejected by validation. Invalid node references detected. All existing graph tests still pass.

### C2: 21 ML Operation Type Test Coverage (FR-005, SC-002)

**Goal**: Add single-node graph CTS tests for the 15 untested ML operation types.

- [x] T067 [P] Add test_single_node_deconvolution (DECONVOLUTION_2D_KHR with ConvolutionKHR descriptor) to tests/cts/test_ml_graph.c
- [x] T068 [P] Add test_single_node_fully_connected (FULLY_CONNECTED_KHR with GemmKHR descriptor) to tests/cts/test_ml_graph.c
- [x] T069 [P] Add test_single_node_average_pool (AVERAGE_POOL_2D_KHR) and test_single_node_global_avg_pool (GLOBAL_AVERAGE_POOL_KHR) with PoolingKHR descriptor to tests/cts/test_ml_graph.c
- [x] T070 [P] Add test_single_node_sigmoid, test_single_node_tanh, test_single_node_leaky_relu, test_single_node_prelu, test_single_node_softmax (5 activation types with ActivationKHR descriptor) to tests/cts/test_ml_graph.c
- [x] T071 [P] Add test_single_node_lrn (LRN_KHR with NormalizationKHR descriptor) to tests/cts/test_ml_graph.c
- [x] T072 [P] Add test_single_node_elementwise_mul (ELEMENTWISE_MUL_KHR with ElementwiseKHR descriptor) to tests/cts/test_ml_graph.c
- [x] T073 [P] Add test_single_node_concat, test_single_node_reshape, test_single_node_transpose, test_single_node_resize (4 descriptor-less operations with pOperationDesc = NULL) to tests/cts/test_ml_graph.c

**Checkpoint**: All 21 ML operation types have explicit single-node graph CTS tests. All tests pass.

### C3: Double-Bind Validation (Constitution V, VUID_BIND_TENSOR_ALREADY_BOUND)

**Goal**: Implement tensor bind validation — reject double-bind, misaligned offset, and null memory.

> **Write tests FIRST — ensure they FAIL before implementation**

- [x] T074 [P] Add test_tensor_double_bind (memoryBound=VK_TRUE → VK_FALSE), test_tensor_bind_misaligned (offset=3 → VK_FALSE), test_tensor_bind_null_memory (memory=VK_NULL_HANDLE → VK_FALSE) to tests/validation/test_vuids.c
- [x] T075 Change vk_ml_validate_tensor_bind signature to accept const struct VkTensorKHR_T *tensor parameter in layers/validation/vk_ml_validation.h
- [x] T076 Implement vk_ml_validate_tensor_bind: check tensor->memoryBound (VUID_BIND_TENSOR_ALREADY_BOUND), pBindInfo->memoryOffset alignment against props->minTensorMemoryAlignment (VUID_BIND_TENSOR_ALIGNMENT), pBindInfo->memory != VK_NULL_HANDLE (VUID_BIND_TENSOR_MEM_TYPE) in layers/validation/tensor_validation.c

**Checkpoint**: Double-bind rejected. Misaligned offsets rejected. Null memory rejected. All existing tensor tests still pass.

### C4: Tensor Barrier Validation (FR-010, FR-011)

**Goal**: Replace tensor_barrier.c placeholder with barrier structure validation. Validate tensor handle, access masks, and queue family indices.

> **Write tests FIRST — ensure they FAIL before implementation**

- [x] T077 [P] Add test_barrier_validation_valid (valid barrier → VK_TRUE), test_barrier_null_tensor (VK_NULL_HANDLE tensor → VK_FALSE), test_barrier_asymmetric_queue_family (one IGNORED, other not → VK_FALSE) to tests/cts/test_synchronization.c
- [x] T078 [P] Add test_barrier_null_tensor_vuid and test_barrier_asymmetric_qf_vuid negative tests to tests/validation/test_vuids.c
- [x] T079 Add vk_ml_validate_tensor_memory_barrier and vk_ml_validate_tensor_dependency_info declarations to layers/validation/vk_ml_validation.h
- [x] T080 Implement vk_ml_validate_tensor_memory_barrier (tensor != VK_NULL_HANDLE, valid access mask bits, symmetric queue family indices) and vk_ml_validate_tensor_dependency_info (barrierCount > 0, per-barrier validation) in src/tensor_barrier.c

**Checkpoint**: Valid barriers accepted. Null tensor rejected. Asymmetric queue family indices rejected. All existing synchronization tests still pass.

---

### Phase 9 Dependencies

```text
All C1-C4 tasks can run in parallel (different files, no mutual dependencies)

C1: T064, T065 (tests) → T066 (implementation)
C2: T067-T073 (all parallel, tests only, no impl changes)
C3: T074 (tests) → T075 (header) → T076 (implementation)
C4: T077, T078 (tests) → T079 (header) → T080 (implementation)
```

### Parallel Example: Phase 9

```bash
# All four issues can be addressed concurrently:
# Developer A: C1 (DAG detection)
Task: "DAG cycle tests in tests/unit/test_dag_validation.c"
Task: "Implement cycle detection in layers/validation/graph_validation.c"

# Developer B: C2 (21-op coverage)
Task: "15 new single-node graph tests in tests/cts/test_ml_graph.c"

# Developer C: C3 (Double-bind)
Task: "Bind validation tests in tests/validation/test_vuids.c"
Task: "Implement bind checks in layers/validation/tensor_validation.c"

# Developer D: C4 (Barriers)
Task: "Barrier validation tests in tests/cts/test_synchronization.c"
Task: "Implement barrier validation in src/tensor_barrier.c"
```

---

## Phase 10: Review Remediation — C1 (Deep-Copy Graph Nodes)

**Purpose**: Fix CRITICAL finding C1 — shallow-copy of `VkMLGraphNodeCreateInfoKHR` nodes leaves dangling pointers to caller-owned `pOperationDesc`, `pInputs`, `pOutputs`, and `pNodeName` memory after `vkCreateMLGraphKHR` returns. Refactor cleanup to use `goto cleanup` pattern (also addresses L4).

**Precondition**: Phase 9 complete. All 12 tests passing.

### Sub-phase 10a: Test-first — write tests that verify deep-copy ownership

- [x] T081 [P] Add `test_graph_node_deep_copy` CTS test to `tests/cts/test_ml_graph.c`: create a graph using stack-local `VkMLPrimitiveDescConvolutionKHR`, `VkMLTensorBindingKHR[]`, and `const char*` node name, then overwrite/zero the stack locals after `vkCreateMLGraphKHR` returns, then call `vkGetMLGraphMemoryRequirementsKHR` and `vkDestroyMLGraphKHR` — verify no crash (proves the graph owns its own copies).

- [x] T082 [P] Add `test_graph_node_null_desc_ops` CTS test to `tests/cts/test_ml_graph.c`: create a graph where nodes use descriptor-less operations (CONCAT, RESHAPE, TRANSPOSE, RESIZE) with `pOperationDesc = NULL` — verify `vkCreateMLGraphKHR` returns `VK_SUCCESS` and destroy succeeds.

- [x] T083 [P] Add `test_graph_node_name_deep_copy` CTS test to `tests/cts/test_ml_graph.c`: create a graph with `pNodeName = "conv2d"` using a local `char[]` buffer, overwrite the buffer after creation, then destroy — verify no crash (proves name was deep-copied).

### Sub-phase 10b: Helper functions — add deep-copy and free utilities

- [x] T084 Add static helper `deep_copy_op_desc(const void *pDesc, const VkAllocationCallbacks *pAllocator)` to `src/ml_graph.c`: switch on `((const VkBaseInStructure *)pDesc)->sType` to determine descriptor struct size, allocate + memcpy. Return NULL for NULL input. Handle all 6 descriptor sType values plus unknown (return NULL, don't fail).

- [x] T085 Add static helper `deep_copy_bindings(const VkMLTensorBindingKHR *pBindings, uint32_t count, const VkAllocationCallbacks *pAllocator)` to `src/ml_graph.c`: allocate `count` bindings, shallow-copy each, set `pNext = NULL`, deep-copy each `pTensorDescription` via existing `deep_copy_tensor_desc()`. On partial failure, free already-copied bindings and return NULL.

- [x] T086 Add static helper `deep_copy_string(const char *str, const VkAllocationCallbacks *pAllocator)` to `src/ml_graph.c`: if `str` is NULL return NULL, else `strlen` + `vk_ml_alloc` + `memcpy` (including null terminator).

- [x] T087 Add static helper `free_node_deep_data(const VkMLGraphNodeCreateInfoKHR *node, const VkAllocationCallbacks *pAllocator)` to `src/ml_graph.c`: free `pOperationDesc`, free each binding's `pTensorDescription` (dims + strides arrays then desc itself) in `pInputs` and `pOutputs` arrays, free the arrays themselves, free `pNodeName`. All frees must tolerate NULL.

### Sub-phase 10c: Refactor vkCreateMLGraphKHR — goto cleanup + deep-copy nodes

- [x] T088 Refactor `vkCreateMLGraphKHR` in `src/ml_graph.c` to use a `goto cleanup` pattern: introduce a `VkResult result` variable and a single `cleanup:` label that frees all partially-allocated graph data based on which fields are non-NULL. Replace the current cascading error blocks (~lines 98-201) with sequential allocation + `if (!ptr) { result = VK_ERROR_OUT_OF_HOST_MEMORY; goto cleanup; }`.

- [x] T089 Replace the `memcpy` of nodes (current lines 98-107) with a loop that, for each node: (1) shallow-copies the `VkMLGraphNodeCreateInfoKHR` struct, (2) sets `pNext = NULL`, (3) calls `deep_copy_op_desc` for `pOperationDesc`, (4) calls `deep_copy_bindings` for `pInputs` (with `inputCount`) and `pOutputs` (with `outputCount`), (5) calls `deep_copy_string` for `pNodeName`. On any allocation failure, `goto cleanup`.

### Sub-phase 10d: Update vkDestroyMLGraphKHR — free deep-copied node data

- [x] T090 Update `vkDestroyMLGraphKHR` in `src/ml_graph.c` to call `free_node_deep_data(&g->nodes[i], pAllocator)` for each node before freeing the `nodes` array itself (line 236).

### Sub-phase 10e: Build + test verification

- [x] T091 Build with `cmake --build build` — zero warnings under `-Wall -Wextra -Wpedantic -Werror`. Run `ctest --output-on-failure` — all tests pass including the 3 new CTS tests. Run `./build/quickstart` — completes successfully.

**Checkpoint**: Graph nodes are fully deep-copied. Caller-owned memory can be freed immediately after `vkCreateMLGraphKHR` without affecting the graph. All tests pass. Cleanup uses `goto` pattern, reducing error-handling code by ~60 lines.

---

### Phase 10 Dependencies

```text
Sub-phase 10a (tests):  T081, T082, T083 — all parallel, no dependencies
Sub-phase 10b (helpers): T084, T085, T086, T087 — all parallel, no dependencies
Sub-phase 10c (refactor): T088 → T089 — sequential (goto pattern first, then node deep-copy)
Sub-phase 10d (destroy):  T090 — depends on T087 (uses free_node_deep_data)
Sub-phase 10e (verify):   T091 — depends on all above

Recommended execution order:
  T081, T082, T083 [P] ← tests first (Constitution Principle IV)
  T084, T085, T086, T087 [P] ← helpers next
  T088 → T089 ← refactor create (sequential)
  T090 ← update destroy
  T091 ← verify
```

---

## Phase 11: Review Remediation — C2 (pNext Chain Preservation)

**Purpose**: Fix CRITICAL finding C2 — `vk_ml_populate_features`, `vk_ml_populate_properties`, and `vk_ml_populate_tensor_format_properties` overwrite caller-set `sType` and `pNext = NULL`, breaking Vulkan pNext structure chaining.

**Precondition**: Phase 10 complete. All 12 tests passing.

### Sub-phase 11a: Test-first — verify pNext chain is preserved after population

- [X] T092 [P] Add `test_features_pnext_preserved` to `tests/cts/test_tensor_formats.c`: set `features.sType` and `features.pNext = &some_chained_struct` before calling `vk_ml_populate_features`, verify `features.pNext` is still `&some_chained_struct` and `features.sType` is unchanged after the call.

- [X] T093 [P] Add `test_properties_pnext_preserved` to `tests/cts/test_tensor_formats.c`: same pattern for `vk_ml_populate_properties` — verify `pNext` and `sType` survive the call.

- [X] T094 [P] Add `test_format_props_pnext_preserved` to `tests/cts/test_tensor_formats.c`: same pattern for `vk_ml_populate_tensor_format_properties` — verify `pNext` and `sType` survive the call.

### Sub-phase 11b: Fix — remove sType/pNext overwrites

- [X] T095 Remove `features->sType = ...` and `features->pNext = NULL` (lines 20-21) from `vk_ml_populate_features` in `src/feature_query.c`.

- [X] T096 Remove `props->sType = ...` and `props->pNext = NULL` (lines 45-46) from `vk_ml_populate_properties` in `src/feature_query.c`.

- [X] T097 Remove `props->sType = ...` and `props->pNext = NULL` (lines 96-97) from `vk_ml_populate_tensor_format_properties` in `src/feature_query.c`.

### Sub-phase 11c: Build + test verification

- [X] T098 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all tests pass including the 3 new pNext preservation tests.

**Checkpoint**: Feature/property queries no longer clobber caller-set `sType`/`pNext`. All pNext chain tests pass. All existing tests still pass.

---

### Phase 11 Dependencies

```text
Sub-phase 11a (tests):  T092, T093, T094 — all parallel
Sub-phase 11b (fix):    T095, T096, T097 — all parallel (same file, different functions)
Sub-phase 11c (verify): T098 — depends on all above

Total: 7 tasks. All test tasks parallel. All fix tasks parallel.
```

---

## Notes

## Phase 12: Review Remediation — H1 (Dangling Description Pointers in Tensor)

**Purpose**: Fix HIGH finding H1 — `vkCreateTensorKHR` shallow-copies `VkTensorDescriptionKHR` leaving `description.pDimensions` and `description.pStrides` as dangling pointers after the caller frees its arrays. `vkGetTensorMemoryRequirementsKHR` has a latent use-after-free fallback path through these stale pointers.

**Precondition**: Phase 11 complete. All 12 tests passing.

### Sub-phase 12a: Test-first — prove tensor owns its description data

- [X] T099 Add `test_tensor_description_owns_dims` to `tests/cts/test_tensor_lifecycle.c`: create a tensor with stack-allocated `dims[]` and `strides[]`, then overwrite the stack arrays with garbage (e.g. `memset(dims, 0xFF, ...)`), call `vkGetTensorMemoryRequirementsKHR`, and verify the returned memory size matches the original dimensions (proving the tensor used its deep-copied data, not the now-corrupted stack). Destroy tensor. Register in `main()`.

### Sub-phase 12b: Fix — redirect description pointers and simplify fallback

- [X] T100 In `vkCreateTensorKHR` in `src/tensor.c`, after the strides deep-copy block (after line 57), add 3 lines: `tensor->description.pDimensions = tensor->dimensions;` `tensor->description.pStrides = tensor->strides;` `tensor->description.pNext = NULL;` — this makes `tensor->description` fully self-consistent with owned data.

- [X] T101 In `vkGetTensorMemoryRequirementsKHR` in `src/tensor.c`, replace `const uint32_t* dims = t->dimensions ? t->dimensions : desc->pDimensions;` (line 109) with `const uint32_t* dims = desc->pDimensions;` — the ternary fallback is now dead code since `desc->pDimensions` always equals `t->dimensions` after the fix above.

### Sub-phase 12c: Build + test verification

- [X] T102 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all tests pass including the new `test_tensor_description_owns_dims` test.

**Checkpoint**: Tensor description pointers are always owned by the tensor object. No dangling pointer paths remain. All existing tests still pass.

---

### Phase 12 Dependencies

```text
Sub-phase 12a (test):   T099 — single test
Sub-phase 12b (fix):    T100, T101 — sequential (T101 depends on T100)
Sub-phase 12c (verify): T102 — depends on all above

Total: 4 tasks. 1 test, 2 code changes (sequential), 1 verification.
```

---

## Phase 13: Review Remediation — H2 (Hardcoded Alignment in vk_ml_alloc)

**Purpose**: Fix HIGH finding H2 — `vk_ml_alloc` passes a hardcoded alignment of `8` to the Vulkan allocation callback. This is insufficient for types requiring stricter alignment and doesn't match `malloc`'s `_Alignof(max_align_t)` guarantee. Replace with `_Alignof(max_align_t)` for portability.

**Precondition**: Phase 12 complete. All 12 tests passing.

### Sub-phase 13a: Test-first — verify allocation callback receives correct alignment

- [X] T103 Add `test_alloc_callback_alignment` to `tests/cts/test_tensor_lifecycle.c`: define a custom `VkAllocationCallbacks` whose `pfnAllocation` captures the `alignment` argument into a static variable, call `vkCreateTensorKHR` with those callbacks, verify the captured alignment is >= `_Alignof(max_align_t)`. Clean up with `vkDestroyTensorKHR`. Register in `main()`.

### Sub-phase 13b: Fix — use _Alignof(max_align_t) in vk_ml_alloc

- [X] T104 In `src/internal.h`, add `#include <stddef.h>` after the existing `#include <string.h>` (line 16) — provides `max_align_t` per C11.

- [X] T105 In `src/internal.h`, in `vk_ml_alloc`, replace the hardcoded `8` (line 89) with `_Alignof(max_align_t)`.

### Sub-phase 13c: Build + test verification

- [X] T106 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all tests pass including the new `test_alloc_callback_alignment` test.

**Checkpoint**: Allocation callback alignment matches `malloc` guarantee. All callers automatically pick up the fix. All existing tests still pass.

---

### Phase 13 Dependencies

```text
Sub-phase 13a (test):   T103 — single test
Sub-phase 13b (fix):    T104, T105 — sequential (T105 depends on T104 for max_align_t)
Sub-phase 13c (verify): T106 — depends on all above

Total: 4 tasks. 1 test, 2 code changes (sequential), 1 verification.
```

---

## Phase 14: Review Remediation — H3 (Integer Overflow in Dimension Product)

**Purpose**: Fix HIGH finding H3 — `uint64_t product` in `vk_ml_validate_tensor_create` can silently wrap when multiplying large dimensions, causing the validation check `product > maxTensorElements` to incorrectly pass. Add an overflow guard before each multiplication.

**Precondition**: Phase 13 complete. All 12 tests passing.

### Sub-phase 14a: Test-first — prove overflow is detected

- [X] T107 Add `test_dimension_product_overflow` to `tests/unit/test_descriptor_validation.c`: create a `VkTensorCreateInfoKHR` with 4 dimensions of 65536 each (product = 2^64, wraps to 0), set `maxTensorElements` to `(1ULL << 32)`, call `vk_ml_validate_tensor_create`, and verify it returns `VK_FALSE`. Register in `main()`.

### Sub-phase 14b: Fix — add overflow guard in dimension loop

- [X] T108 In `vk_ml_validate_tensor_create` in `layers/validation/tensor_validation.c`, inside the dimension loop (after the per-dimension bounds check on line 37-38), add an overflow guard before `product *= desc->pDimensions[i]` (line 39): `if (product > props->maxTensorElements / desc->pDimensions[i]) return VK_FALSE;` — this rejects dimensions whose cumulative product would exceed `maxTensorElements` without ever overflowing.

### Sub-phase 14c: Build + test verification

- [X] T109 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all tests pass including the new `test_dimension_product_overflow` test.

**Checkpoint**: Dimension product overflow is caught before it occurs. Validation correctly rejects tensors whose total element count exceeds `maxTensorElements`. All existing tests still pass.

---

### Phase 14 Dependencies

```text
Sub-phase 14a (test):   T107 — single test
Sub-phase 14b (fix):    T108 — single change
Sub-phase 14c (verify): T109 — depends on all above

Total: 3 tasks. 1 test, 1 code change, 1 verification.
```

---

## Phase 15: Review Remediation — H4 (NULL pNodes with nodeCount > 0)

**Purpose**: Fix HIGH finding H4 — `vk_ml_validate_graph_create` accepts `pNodes == NULL` when `nodeCount > 0`, skipping DFS cycle detection entirely. Downstream `vkCreateMLGraphKHR` will dereference the NULL pointer and crash.

**Precondition**: Phase 14 complete. All 12 tests passing.

### Sub-phase 15a: Test-first — prove NULL pNodes is rejected

- [X] T110 [P] Add `test_null_pnodes_with_nodecount` to `tests/unit/test_dag_validation.c`: set `nodeCount = 1` and `pNodes = NULL` in a `VkMLGraphCreateInfoKHR`, call `vk_ml_validate_graph_create`, and verify it returns `VK_FALSE`. Register in `main()`.

### Sub-phase 15b: Fix — add null guard before DFS block

- [X] T111 In `vk_ml_validate_graph_create` in `layers/validation/graph_validation.c`, add `if (!pCreateInfo->pNodes) return VK_FALSE;` before the DFS block (before line 62). Since `nodeCount > 0` is guaranteed at this point (line 50 rejects 0), a NULL `pNodes` pointer is invalid.

### Sub-phase 15c: Build + test verification

- [X] T112 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all tests pass including the new `test_null_pnodes_with_nodecount` test.

**Checkpoint**: NULL `pNodes` with `nodeCount > 0` is correctly rejected. No null-dereference path remains. All existing tests still pass.

---

### Phase 15 Dependencies

```text
Sub-phase 15a (test):   T110 — single test
Sub-phase 15b (fix):    T111 — single change
Sub-phase 15c (verify): T112 — depends on all above

Total: 3 tasks. 1 test, 1 code change, 1 verification.
```

---

## Phase 16: Review Remediation — H5 (Dispatch Validation NULL Array Pointers)

**Purpose**: Fix HIGH finding H5 — `vk_ml_validate_dispatch` doesn't check that `pInputTensors`, `pOutputTensors`, and `pWeightTensors` are non-NULL when their respective counts are > 0. Passing NULL arrays with positive counts will cause downstream NULL dereferences.

**Precondition**: Phase 15 complete. All 12 tests passing.

### Sub-phase 16a: Test-first — prove NULL array pointers are rejected

- [X] T113 Add `test_dispatch_null_input_tensors` to `tests/cts/test_ml_dispatch.c`: set `inputTensorCount = 1` and `pInputTensors = NULL`, call `vk_ml_validate_dispatch`, verify it returns `VK_FALSE`. Register in `main()`.

- [X] T114 Add `test_dispatch_null_output_tensors` to `tests/cts/test_ml_dispatch.c`: set `outputTensorCount = 1` and `pOutputTensors = NULL`, call `vk_ml_validate_dispatch`, verify it returns `VK_FALSE`. Register in `main()`.

- [X] T115 Add `test_dispatch_null_weight_tensors` to `tests/cts/test_ml_dispatch.c`: set `weightTensorCount = 1` and `pWeightTensors = NULL`, call `vk_ml_validate_dispatch`, verify it returns `VK_FALSE`. Register in `main()`.

### Sub-phase 16b: Fix — add NULL array guards

- [X] T116 In `vk_ml_validate_dispatch` in `layers/validation/dispatch_validation.c`, after the count checks (line 23), add 3 NULL-pointer guards: `if (!pDispatchInfo->pInputTensors) return VK_FALSE;` `if (!pDispatchInfo->pOutputTensors) return VK_FALSE;` `if (pDispatchInfo->weightTensorCount > 0 && !pDispatchInfo->pWeightTensors) return VK_FALSE;`

### Sub-phase 16c: Build + test verification

- [X] T117 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all tests pass including the 3 new dispatch NULL-pointer tests.

**Checkpoint**: Dispatch validation rejects NULL tensor arrays when counts are positive. No NULL-dereference paths remain. All existing tests still pass.

---

### Phase 16 Dependencies

```text
Sub-phase 16a (tests):  T113, T114, T115 — all in same file, sequential
Sub-phase 16b (fix):    T116 — single change
Sub-phase 16c (verify): T117 — depends on all above

Total: 5 tasks. 3 tests, 1 code change, 1 verification.
```

---

## Phase 17: Review Remediation — H6 (Move Barrier Validation to Correct Directory)

**Purpose**: Fix HIGH finding H6 — `vk_ml_validate_tensor_memory_barrier` and `vk_ml_validate_tensor_dependency_info` are validation functions living in `src/tensor_barrier.c` instead of `layers/validation/`. Move them to the validation layer so they compile against `vk_ml_validation.h` and signatures stay in sync.

**Precondition**: Phase 16 complete. All 12 tests passing.

### Sub-phase 17a: Create new barrier validation file

- [X] T118 Create `layers/validation/barrier_validation.c` with the full contents of `src/tensor_barrier.c`, but replace `#include <vulkan/vulkan_ml_primitives.h>` with `#include "../validation/vk_ml_validation.h"` (which transitively includes the public header). Keep the `VALID_TENSOR_ACCESS_MASK` constant and both validation functions unchanged.

### Sub-phase 17b: Update CMakeLists.txt

- [X] T119 In `CMakeLists.txt`, remove `src/tensor_barrier.c` from the `IMPL_SOURCES` list.

- [X] T120 In `CMakeLists.txt`, add `layers/validation/barrier_validation.c` to the `VALIDATION_SOURCES` list.

### Sub-phase 17c: Delete old file

- [X] T121 Delete `src/tensor_barrier.c`.

### Sub-phase 17d: Build + test verification

- [X] T122 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 12 tests pass (barrier VUID tests in `test_vuids.c` still link correctly from the validation library).

**Checkpoint**: Barrier validation now lives in `layers/validation/` alongside all other validation code. Includes `vk_ml_validation.h` for compile-time signature checking. All tests still pass.

---

### Phase 17 Dependencies

```text
Sub-phase 17a (create):  T118 — create new file
Sub-phase 17b (cmake):   T119, T120 — sequential (same file)
Sub-phase 17c (delete):  T121 — depends on T118-T120
Sub-phase 17d (verify):  T122 — depends on all above

Total: 5 tasks. All sequential.
```

---

## Phase 18: Review Remediation — H7 (Create README.md)

**Goal**: Create a comprehensive `README.md` at the repository root covering project description, build instructions, test instructions, directory structure, extension status, and link to the spec. Resolves HIGH finding H7 from `review-findings.md`.

### Sub-phase 18a: Create README

- [X] T123 Create `README.md` at the repository root with the following sections (all content derived from existing project artifacts):

  1. **Title**: `VK_KHR_ml_primitives` with one-line tagline describing it as a Vulkan extension for GPU-accelerated ML.
  2. **Overview**: One paragraph summarizing the extension — 4 new object types (`VkTensorKHR`, `VkTensorViewKHR`, `VkMLGraphKHR`, `VkMLSessionKHR`), 21 ML primitive operations, compiled graph dispatch, Vulkan 1.3 baseline.
  3. **Extension Dependencies**: Bullet list — `VK_KHR_cooperative_matrix`, `VK_KHR_timeline_semaphore`, `VK_KHR_maintenance5`, `VK_KHR_format_feature_flags2`, `SPV_KHR_tensor`.
  4. **Repository Structure**: Tree diagram matching actual on-disk layout:
     - `include/vulkan/vulkan_ml_primitives.h` — public C header
     - `src/` — reference ICD implementation (8 source files + `internal.h`)
     - `layers/validation/` — validation layer (5 source files + `vk_ml_validation.h`)
     - `tests/cts/` — 9 conformance test suites
     - `tests/validation/` — VUID negative tests (`test_vuids.c`)
     - `tests/unit/` — 2 unit test suites
     - `examples/` — quickstart example
     - `spec/` — authoritative `.adoc` specification
     - `specs/` — design artifacts (plan, tasks, review findings)
  5. **Prerequisites**: CMake 3.20+, Vulkan SDK 1.3+, C11 compiler (GCC 11+ / Clang 14+ / MSVC 2022+).
  6. **Building**: `cmake -B build -S .` then `cmake --build build`.
  7. **Running Tests**: `cd build && ctest --output-on-failure` — 12 test suites covering tensor lifecycle, views, copies, formats, graphs, sessions, dispatch, synchronization, SPIR-V, VUIDs, DAG validation, and descriptor validation.
  8. **Quick Start**: Reference `examples/quickstart.c` with a brief summary of the workflow (create tensors → build graph → create session → dispatch → cleanup).
  9. **Static Analysis**: Note that `clang-tidy` is auto-integrated via CMake (`CMAKE_C_CLANG_TIDY`); optional `cppcheck` target available via `cmake --build build --target cppcheck`.
  10. **Specification**: Link to `spec/VK_KHR_ml_primitives.adoc`. Note extension revision 1, cross-vendor KHR status.
  11. **License**: Placeholder section noting the license is TBD.

### Sub-phase 18b: Verify README accuracy

- [X] T124 Verify the README is accurate: (a) all directories/files mentioned in the tree diagram exist on disk, (b) the build commands (`cmake -B build -S . && cmake --build build`) succeed with zero warnings, (c) the test command (`cd build && ctest --output-on-failure`) runs 12 tests and all pass, (d) `examples/quickstart.c` is correctly referenced.

**Checkpoint**: Repository has a comprehensive README. All referenced paths, commands, and counts are verified accurate.

---

### Phase 18 Dependencies

```text
Sub-phase 18a (create):  T123 — create README.md
Sub-phase 18b (verify):  T124 — depends on T123

Total: 2 tasks. Sequential.
```

---

## Phase 19: Review Remediation — H8 (OOM / Allocation-Failure Tests)

**Goal**: Add CTS tests exercising every `VK_ERROR_OUT_OF_HOST_MEMORY` return path in the four `vkCreate*` functions. Uses a custom `VkAllocationCallbacks` that returns `NULL` after N allocations to verify: (a) correct `VkResult`, (b) no partial-allocation leaks, (c) no partial object returned. Resolves HIGH finding H8 from `review-findings.md`.

### Sub-phase 19a: Create OOM test suite

- [X] T125 Create `tests/cts/test_oom.c` with the following contents:

  1. **Failing allocator infrastructure**:
     - Define `FailingAllocatorData` struct with fields: `uint32_t alloc_count` (incremented on each `pfnAllocation` call), `uint32_t fail_after` (return `NULL` when `alloc_count > fail_after`), `uint32_t free_count` (incremented on each `pfnFree` call).
     - Implement `failing_alloc(void *pUserData, size_t size, size_t alignment, VkSystemAllocationScope scope)`: increment `alloc_count`; if `alloc_count > fail_after`, return `NULL`; otherwise call `aligned_alloc`/`malloc` normally.
     - Implement `failing_free(void *pUserData, void *pMemory)`: if `pMemory` is non-NULL, increment `free_count` and call `free`.
     - Implement `failing_realloc` as a stub that returns `NULL` (not used by implementation).
     - Helper `make_failing_allocator(FailingAllocatorData *data)` returns a `VkAllocationCallbacks` struct wired to the above functions with `pUserData = data`.
     - Helper `reset_failing_allocator(FailingAllocatorData *data, uint32_t fail_after)` zeroes all counters and sets `fail_after`.

  2. **`test_tensor_create_oom`**: Set up a valid `VkTensorCreateInfoKHR` with `dimensionCount = 2`, `pDimensions`, `pStrides`, and `queueFamilyIndexCount = 1`. Loop `fail_after` from `0` to `3` (4 allocation points in `vkCreateTensorKHR`). For each iteration: reset the allocator, call `vkCreateTensorKHR`, assert return is `VK_ERROR_OUT_OF_HOST_MEMORY`, assert `free_count == alloc_count - 1` (all successful allocations freed), assert `pTensor` output handle was NOT modified from its sentinel value.

  3. **`test_tensor_view_create_oom`**: Set up a valid `VkTensorViewCreateInfoKHR` with `dimensionCount = 2`, `pDimensionOffsets`, `pDimensionSizes`. Loop `fail_after` from `0` to `2` (3 allocation points in `vkCreateTensorViewKHR`). For each iteration: reset the allocator, call `vkCreateTensorViewKHR`, assert return is `VK_ERROR_OUT_OF_HOST_MEMORY`, assert `free_count == alloc_count - 1`, assert `pView` output handle was NOT modified.

  4. **`test_graph_create_oom`**: Set up a valid `VkMLGraphCreateInfoKHR` with 1 node containing: `operationType = VK_ML_OPERATION_TYPE_ACTIVATION_KHR`, a valid `VkMLPrimitiveDescActivationKHR` as `pOperationDesc`, 1 input binding with a `VkTensorDescriptionKHR` (2 dims), 1 output binding with a `VkTensorDescriptionKHR` (2 dims), a `pNodeName`, plus 1 external input desc, 1 external output desc, 0 constant weights. Loop `fail_after` from `0` upward: reset allocator, call `vkCreateMLGraphKHR`; if result is `VK_SUCCESS`, destroy the graph and break (we've covered all OOM paths); otherwise assert result is `VK_ERROR_OUT_OF_HOST_MEMORY` and assert `free_count == alloc_count - 1`. Assert that the loop ran at least 5 iterations (proving multiple OOM points were exercised).

  5. **`test_session_create_oom`**: Set up a valid `VkMLSessionCreateInfoKHR`. Set `fail_after = 0`. Call `vkCreateMLSessionKHR`, assert return is `VK_ERROR_OUT_OF_HOST_MEMORY`, assert `free_count == 0` (the single allocation failed, nothing to free), assert output handle was NOT modified.

  6. **`main`** function: call all 4 test functions, print results, return 0 on all pass or 1 on failure.

  Include headers: `<vulkan/vulkan_ml_primitives.h>`, `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<stdint.h>`.

### Sub-phase 19b: Update CMakeLists.txt

- [X] T126 In `CMakeLists.txt`, add `tests/cts/test_oom.c` to the `CTS_TESTS` list (after `test_spirv_tensor.c`).

### Sub-phase 19c: Build + test verification

- [X] T127 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass. Verify the new `cts_test_oom` test appears and passes.

**Checkpoint**: All four `vkCreate*` functions have OOM tests. Every allocation failure point returns `VK_ERROR_OUT_OF_HOST_MEMORY` without leaking partially-allocated resources.

---

### Phase 19 Dependencies

```text
Sub-phase 19a (test):    T125 — create test file
Sub-phase 19b (cmake):   T126 — depends on T125
Sub-phase 19c (verify):  T127 — depends on T125, T126

Total: 3 tasks. Sequential.
```

---

## Phase 20: Review Remediation — H9 (Strengthen Tautological Tests)

**Goal**: Replace or strengthen 7 tautological tests that always pass regardless of implementation correctness. Each test must now exercise real API behavior or validation logic. Resolves HIGH finding H9 from `review-findings.md`.

### Sub-phase 20a: Strengthen null-handle destroy tests

- [X] T128 [P] In `tests/cts/test_tensor_lifecycle.c`, replace the body of `test_destroy_null_handle` (line 177-181). Instead of unconditional `return 0`:
  1. Create a valid tensor (`vkCreateTensorKHR`) and store the handle.
  2. Call `vkDestroyTensorKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, NULL)` — the null-handle destroy.
  3. Verify the live tensor is still valid: cast to `VkTensorKHR_T*` and assert `description.dimensionCount` still holds its expected value.
  4. Destroy the live tensor normally — assert it doesn't crash.
  5. Return 0 only if all assertions pass, 1 otherwise.

- [X] T129 [P] In `tests/cts/test_tensor_view.c`, replace the body of `test_destroy_view_null` (line 118-122). Instead of unconditional `return 0`:
  1. Create a valid tensor and a valid tensor view.
  2. Call `vkDestroyTensorViewKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, NULL)` — the null-handle destroy.
  3. Verify the live view is still valid: cast to `VkTensorViewKHR_T*` and assert `format` still holds its expected value.
  4. Destroy the view and tensor normally.
  5. Return 0 only if all assertions pass, 1 otherwise.
  6. Add `#include "../../src/internal.h"` at the top if not already present (needed for `VkTensorViewKHR_T`).

### Sub-phase 20b: Strengthen tensor copy tests

- [X] T130 [P] In `tests/cts/test_tensor_copy.c`, strengthen both copy tests:

  **`test_copy_basic` (line 26)**: After the `vkCmdCopyTensorKHR` call, add assertions:
  1. Verify `src` tensor is still valid (cast to `VkTensorKHR_T*`, check `description.dimensionCount == 4`).
  2. Verify `dst` tensor is still valid (same check).
  3. Return 0 only if both assertions pass.

  **`test_copy_null_cmd` (line 89)**: Add assertions:
  1. After the copy call with `VK_NULL_HANDLE` command buffer, verify both tensors are still valid (same internal state check).
  2. Also add a test for `pCopyInfo = NULL`: call `vkCmdCopyTensorKHR(cmd, NULL)` and verify the tensors remain valid.
  3. Return 0 only if all assertions pass.

  Add `#include "../../src/internal.h"` at the top if not already present (needed for `VkTensorKHR_T`).

### Sub-phase 20c: Replace struct-readback tests with validation tests

- [X] T131 In `tests/cts/test_synchronization.c`, replace the bodies of three tautological tests with validation-layer calls. The file already includes `../../layers/validation/vk_ml_validation.h` (from prior work). Replace as follows:

  **`test_barrier_structure` (line 30)**: Replace C struct readback with:
  1. Create a valid tensor via `vkCreateTensorKHR`.
  2. Build a valid `VkTensorMemoryBarrierKHR` with the tensor handle and valid access masks (`VK_ACCESS_2_ML_GRAPH_READ_BIT_KHR` src, `VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR` dst), both queue families `VK_QUEUE_FAMILY_IGNORED`.
  3. Call `vk_ml_validate_tensor_memory_barrier(&barrier)` and assert it returns `VK_TRUE`.
  4. Set `barrier.tensor = VK_NULL_HANDLE` and assert validation returns `VK_FALSE`.
  5. Destroy the tensor. Return 0 if all pass.

  **`test_dependency_info` (line 62)**: Replace C struct readback with:
  1. Create a valid tensor.
  2. Build a valid barrier and dependency info (`tensorMemoryBarrierCount = 1`, `pTensorMemoryBarriers = &barrier`).
  3. Assert `vk_ml_validate_tensor_dependency_info(&depInfo)` returns `VK_TRUE`.
  4. Set `depInfo.tensorMemoryBarrierCount = 0` and assert validation returns `VK_FALSE`.
  5. Restore count, set `depInfo.pTensorMemoryBarriers = NULL` and assert validation returns `VK_FALSE`.
  6. Destroy the tensor. Return 0 if all pass.

  **`test_queue_family_transfer` (line 268)**: Replace C struct readback with:
  1. Create a valid tensor.
  2. Build a barrier with `srcQueueFamilyIndex = 0`, `dstQueueFamilyIndex = 1` (both explicit — valid symmetric transfer).
  3. Assert `vk_ml_validate_tensor_memory_barrier(&barrier)` returns `VK_TRUE`.
  4. Set `srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED` (asymmetric — one IGNORED, one not).
  5. Assert validation returns `VK_FALSE`.
  6. Set both to `VK_QUEUE_FAMILY_IGNORED` (symmetric — valid).
  7. Assert validation returns `VK_TRUE`.
  8. Destroy the tensor. Return 0 if all pass.

### Sub-phase 20d: Build + test verification

- [X] T132 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass. Verify no test is tautological: each modified test has at least one real assertion that could fail if the implementation were broken.

**Checkpoint**: All 7 tautological tests now exercise real API behavior or validation logic. No test unconditionally returns 0.

---

### Phase 20 Dependencies

```text
Sub-phase 20a (destroy):  T128, T129 — [P] (different files)
Sub-phase 20b (copy):     T130 — [P] (different file)
Sub-phase 20c (sync):     T131 — separate file
Sub-phase 20d (verify):   T132 — depends on T128-T131

Total: 5 tasks. T128-T130 parallelizable, T131 sequential, T132 final.
```

---

## Phase 21: Review Remediation — H10 (Remove Internal Representation Dependencies from CTS)

**Goal**: Replace all internal struct casts (`VkTensorKHR_T*`, `VkTensorViewKHR_T*`, `VkMLSessionKHR_T*`) in CTS tests with public API queries. Remove `internal.h` includes from CTS files where no longer needed. Resolves HIGH finding H10 from `review-findings.md`.

### Sub-phase 21a: Fix `test_tensor_lifecycle.c` (2 casts)

- [X] T133 In `tests/cts/test_tensor_lifecycle.c`, replace the internal cast in `test_bind_tensor_memory` (around line 171). Currently casts to `VkTensorKHR_T*` and checks `t->memoryBound == VK_TRUE`. Replace with: after `vkBindTensorMemoryKHR` returns `VK_SUCCESS`, call `vkGetTensorMemoryRequirementsKHR` on the tensor and assert the returned `memoryRequirements.size > 0`. This proves the tensor is still valid and functional post-bind. The `VK_SUCCESS` return from bind is the observable contract for bind succeeding.

- [X] T134 In `tests/cts/test_tensor_lifecycle.c`, replace the internal cast in `test_destroy_null_handle` (around line 205). Currently casts to `VkTensorKHR_T*` and checks `t->description.dimensionCount`. Replace with: call `vkGetTensorMemoryRequirementsKHR` on the live tensor and assert `memoryRequirements.size > 0`. This proves the live tensor survived the null-handle destroy intact.

- [X] T135 In `tests/cts/test_tensor_lifecycle.c`, remove `#include "internal.h"` (line 10) since no internal types are used after T133-T134. The file already includes `<vulkan/vulkan_ml_primitives.h>` which provides all needed public types.

### Sub-phase 21b: Fix `test_tensor_view.c` (1 cast)

- [X] T136 [P] In `tests/cts/test_tensor_view.c`, replace the internal cast in `test_destroy_view_null` (around line 165). Currently casts to `VkTensorViewKHR_T*` and checks `v->format`. Replace with: call `vkGetTensorMemoryRequirementsKHR` on the parent tensor and assert `memoryRequirements.size > 0` (proves parent tensor survived). Assert `view != VK_NULL_HANDLE` (proves handle is still set). Remove `#include "../../src/internal.h"` (line 10).

### Sub-phase 21c: Fix `test_tensor_copy.c` (4 casts)

- [X] T137 [P] In `tests/cts/test_tensor_copy.c`, replace all 4 internal casts:

  **`test_copy_basic`** (around lines 84-85): Currently casts `src` and `dst` to `VkTensorKHR_T*` and checks `description.dimensionCount == 4`. Replace with: build a `VkTensorMemoryRequirementsInfoKHR` for each tensor, call `vkGetTensorMemoryRequirementsKHR`, assert returned `size > 0` for both `src` and `dst`.

  **`test_copy_null_cmd`** (around lines 151-152): Same replacement — query memory requirements for both tensors and assert `size > 0`.

  Remove `#include "../../src/internal.h"` (line 10).

### Sub-phase 21d: Fix `test_ml_session.c` (1 cast)

- [X] T138 [P] In `tests/cts/test_ml_session.c`, replace the internal cast in `test_session_auto_alloc` (around line 192). Currently casts to `VkMLSessionKHR_T*` and checks `s->autoAllocated == VK_TRUE`. Replace with: the session was created with `scratchMemory == VK_NULL_HANDLE` and returned `VK_SUCCESS`. Assert `session != VK_NULL_HANDLE` (already done by the existing check on line 187). The auto-allocation flag is an internal implementation detail with no observable public API effect. Remove the cast entirely. Remove `#include "../../src/internal.h"` (line 7).

### Sub-phase 21e: Build + test verification

- [X] T139 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass. Verify: `grep -r 'VkTensorKHR_T\|VkTensorViewKHR_T\|VkMLSessionKHR_T\|VkMLGraphKHR_T' tests/cts/` returns zero matches (no internal type references remain in CTS).

**Checkpoint**: All CTS tests use only the public API surface. No `internal.h` includes or internal struct casts remain in `tests/cts/`. Tests are portable across any conformant ICD.

---

### Phase 21 Dependencies

```text
Sub-phase 21a (lifecycle): T133, T134, T135 — sequential (same file)
Sub-phase 21b (view):      T136 — [P] (different file)
Sub-phase 21c (copy):      T137 — [P] (different file)
Sub-phase 21d (session):   T138 — [P] (different file)
Sub-phase 21e (verify):    T139 — depends on T133-T138

Total: 7 tasks. T136, T137, T138 parallelizable. T133-T135 sequential. T139 final.
```

---

## Phase 22: Review Remediation — M1 (Wrong Error Code for NULL Parameter Validation)

**Goal**: Replace `VK_ERROR_INITIALIZATION_FAILED` with `VK_ERROR_UNKNOWN` in all ICD-level NULL parameter guards. `VK_ERROR_INITIALIZATION_FAILED` is semantically reserved for driver/device initialization failures. `VK_ERROR_UNKNOWN` (added in Vulkan 1.2) is the correct catch-all for unexpected conditions that don't match any specific error code. Resolves MEDIUM finding M1 from `review-findings.md`.

### Sub-phase 22a: Fix create functions (4 files, parallelizable)

- [X] T140 [P] In `src/tensor.c`, replace `VK_ERROR_INITIALIZATION_FAILED` on line 20 (the `!pCreateInfo || !pCreateInfo->pDescription || !pTensor` guard in `vkCreateTensorKHR`) with `VK_ERROR_UNKNOWN`. Also replace the same error code on line 135 (the `!pBindInfos` guard in `vkBindTensorMemoryKHR`) with `VK_ERROR_UNKNOWN`.

- [X] T141 [P] In `src/tensor_view.c`, replace `VK_ERROR_INITIALIZATION_FAILED` on line 20 (the `!pCreateInfo || !pView` guard in `vkCreateTensorViewKHR`) with `VK_ERROR_UNKNOWN`.

- [X] T142 [P] In `src/ml_graph.c`, replace `VK_ERROR_INITIALIZATION_FAILED` on line 239 (the `!pCreateInfo || !pGraph` guard in `vkCreateMLGraphKHR`) with `VK_ERROR_UNKNOWN`.

- [X] T143 [P] In `src/ml_session.c`, replace `VK_ERROR_INITIALIZATION_FAILED` on line 20 (the `!pCreateInfo || !pSession` guard in `vkCreateMLSessionKHR`) with `VK_ERROR_UNKNOWN`.

### Sub-phase 22b: Build + test verification

- [X] T144 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass. Verify: `grep -rn 'VK_ERROR_INITIALIZATION_FAILED' src/` returns zero matches (no misused error code remains in ICD sources).

**Checkpoint**: All ICD-level NULL parameter guards return `VK_ERROR_UNKNOWN`. No test breakage (zero tests asserted on the old error code). Semantically correct per Vulkan convention.

---

### Phase 22 Dependencies

```text
Sub-phase 22a (create fns):  T140, T141, T142, T143 — [P] (different files)
Sub-phase 22b (verify):      T144 — depends on T140-T143

Total: 5 tasks. T140-T143 fully parallelizable. T144 final.
```

---

## Phase 23: Review Remediation — M2 (No Double-Bind Protection in vkBindTensorMemoryKHR)

**Goal**: Add an ICD-level guard in `vkBindTensorMemoryKHR` to reject re-binding a tensor that already has memory bound. The validation layer already checks `VUID_BIND_TENSOR_ALREADY_BOUND`, but the ICD silently overwrites on re-bind if the validation layer is disabled. Resolves MEDIUM finding M2 from `review-findings.md`.

### Sub-phase 23a: Add double-bind guard

- [X] T145 In `src/tensor.c`, inside the `vkBindTensorMemoryKHR` bind loop (after line 142: `VkTensorKHR_T* t = (VkTensorKHR_T*)(uintptr_t)info->tensor;`), add `if (t->memoryBound) return VK_ERROR_UNKNOWN;` before the lines that overwrite `boundMemory`, `memoryOffset`, and `memoryBound`. This prevents silent re-binding when the validation layer is disabled.

### Sub-phase 23b: Build + test verification

- [X] T146 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass. The existing `test_tensor_double_bind` in `tests/validation/test_vuids.c` exercises the validation layer path; the ICD guard is an independent safety net.

**Checkpoint**: `vkBindTensorMemoryKHR` now rejects double-bind at the ICD level with `VK_ERROR_UNKNOWN`. Validation layer continues to provide the detailed VUID diagnostic. All 13 tests pass.

---

### Phase 23 Dependencies

```text
Sub-phase 23a (guard):   T145 — single file change
Sub-phase 23b (verify):  T146 — depends on T145

Total: 2 tasks. Sequential.
```

---

## Phase 24: Review Remediation — M3 (No Alignment Validation in vkBindTensorMemoryKHR)

**Goal**: Add an ICD-level alignment check for `memoryOffset` in `vkBindTensorMemoryKHR` using the reference implementation's own `VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN` constant (64). The validation layer already checks `VUID_BIND_TENSOR_ALIGNMENT` against `props->minTensorMemoryAlignment`, but the ICD silently accepts misaligned offsets if the validation layer is disabled. Resolves MEDIUM finding M3 from `review-findings.md`.

### Sub-phase 24a: Add alignment guard

- [X] T147 In `src/tensor.c`, inside the `vkBindTensorMemoryKHR` bind loop, after the `if (t->memoryBound) return VK_ERROR_UNKNOWN;` guard (line 144) and before `t->boundMemory = info->memory;` (line 145), add: `if (VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN > 0 && info->memoryOffset % VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN != 0) return VK_ERROR_UNKNOWN;`. This rejects misaligned offsets at the ICD level.

### Sub-phase 24b: Build + test verification

- [X] T148 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: `vkBindTensorMemoryKHR` now rejects misaligned `memoryOffset` at the ICD level. Validation layer provides the VUID diagnostic; ICD provides the safety net. All 13 tests pass.

---

### Phase 24 Dependencies

```text
Sub-phase 24a (guard):   T147 — single file change
Sub-phase 24b (verify):  T148 — depends on T147

Total: 2 tasks. Sequential.
```

---

## Phase 25: Review Remediation — M4 (Magic Integer Literals in Format Element Size)

**Goal**: Replace raw integer literals in `vk_ml_format_element_size` switch cases with their corresponding `VK_FORMAT_*` enum names. The enum values are defined in `include/vulkan/vulkan_ml_primitives.h:111-114`. Resolves MEDIUM finding M4 from `review-findings.md`.

### Sub-phase 25a: Replace magic integers

- [X] T149 In `src/internal.h`, replace the 4 magic integer cases in `vk_ml_format_element_size` (lines 123-126):
  - `case 1000559001: /* VK_FORMAT_R16_BFLOAT_KHR */ return 2;` → `case (uint32_t)VK_FORMAT_R16_BFLOAT_KHR: return 2;`
  - `case 1000559002: /* VK_FORMAT_R8_E4M3_KHR */   return 1;` → `case (uint32_t)VK_FORMAT_R8_E4M3_KHR:   return 1;`
  - `case 1000559003: /* VK_FORMAT_R8_E5M2_KHR */   return 1;` → `case (uint32_t)VK_FORMAT_R8_E5M2_KHR:   return 1;`
  - `case 1000559000: /* VK_FORMAT_R8_BOOL */       return 1;` → `case (uint32_t)VK_FORMAT_R8_BOOL:        return 1;`

### Sub-phase 25b: Build + test verification

- [X] T150 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: No magic integer literals remain in `vk_ml_format_element_size`. All format cases use named enum constants. Zero behavioral change.

---

### Phase 25 Dependencies

```text
Sub-phase 25a (replace): T149 — single file change
Sub-phase 25b (verify):  T150 — depends on T149

Total: 2 tasks. Sequential.
```

---

## Phase 26: Review Remediation — M5 (is_finite_float Fragile Under -ffast-math)

**Goal**: Replace the hand-rolled `is_finite_float` function with C11 standard `isfinite()` from `<math.h>`. The current implementation `(f == f) && (f - f == 0.0f)` relies on IEEE 754 semantics and is optimized away to `true` under `-ffast-math`. Resolves MEDIUM finding M5 from `review-findings.md`.

### Sub-phase 26a: Replace function and add link dependency

- [X] T151 In `src/ml_primitives.c`, add `#include <math.h>` after the existing `#include "internal.h"` (line 6). Then replace the body of `is_finite_float` (line 14: `return (f == f) && (f - f == 0.0f);`) with `return isfinite(f);`.

- [X] T152 In `CMakeLists.txt`, on line 41 (`target_link_libraries(vk_ml_primitives PUBLIC Vulkan::Vulkan)`), append `m` to the link libraries for non-MSVC platforms. Use a generator expression: `target_link_libraries(vk_ml_primitives PUBLIC Vulkan::Vulkan $<$<NOT:$<C_COMPILER_ID:MSVC>>:m>)`. This links libm on GCC/Clang platforms where `isfinite()` may require it, and skips it on MSVC where math functions are in the CRT.

### Sub-phase 26b: Build + test verification

- [X] T153 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: `is_finite_float` now uses the standards-conformant `isfinite()` macro. Immune to `-ffast-math` optimization. All 13 tests pass.

---

### Phase 26 Dependencies

```text
Sub-phase 26a (replace): T151, T152 — [P] (different files)
Sub-phase 26b (verify):  T153 — depends on T151, T152

Total: 3 tasks. T151 and T152 parallelizable. T153 final.
```

---

## Phase 27: Review Remediation — M6 (Inconsistent Include Pattern in Impl Files)

**Goal**: Replace `#include <vulkan/vulkan_ml_primitives.h>` with `#include "internal.h"` in the 2 remaining implementation files that use the public header directly. All other `src/*.c` files already use `"internal.h"`. Resolves MEDIUM finding M6 from `review-findings.md`.

### Sub-phase 27a: Fix includes (parallelizable)

- [X] T154 [P] In `src/tensor_copy.c`, replace `#include <vulkan/vulkan_ml_primitives.h>` (line 9) with `#include "internal.h"`.

- [X] T155 [P] In `src/ml_dispatch.c`, replace `#include <vulkan/vulkan_ml_primitives.h>` (line 9) with `#include "internal.h"`.

### Sub-phase 27b: Build + test verification

- [X] T156 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass. Verify: `grep -rn 'vulkan_ml_primitives.h' src/` returns zero matches (all impl files now use `internal.h`).

**Checkpoint**: All `src/*.c` files consistently include `"internal.h"`. No direct public header includes remain in implementation code. All 13 tests pass.

---

### Phase 27 Dependencies

```text
Sub-phase 27a (includes): T154, T155 — [P] (different files)
Sub-phase 27b (verify):   T156 — depends on T154, T155

Total: 3 tasks. T154, T155 parallelizable. T156 final.
```

---

## Phase 28: Review Remediation — M7 (No sType Validation Anywhere)

**Goal**: Add `sType` validation as the first check (after NULL guard) in all 14 validation functions across 5 validation files. Wrong `sType` is a common application error and standard Vulkan validation practice. Resolves MEDIUM finding M7 from `review-findings.md`.

### Sub-phase 28a: Tensor validation (4 functions)

- [X] T157 In `layers/validation/tensor_validation.c`, add sType checks to 4 functions:
  - `vk_ml_validate_tensor_create`: after `if (!pCreateInfo || ...) return VK_FALSE;` (line 17), add `if (pCreateInfo->sType != VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR) return VK_FALSE;`
  - `vk_ml_validate_tensor_view_create`: after NULL guard (line 70), add `if (pCreateInfo->sType != VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR) return VK_FALSE;`
  - `vk_ml_validate_tensor_bind`: after NULL guard (line 99), add `if (pBindInfo->sType != VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR) return VK_FALSE;`
  - `vk_ml_validate_tensor_copy`: after NULL guard (line 123), add `if (pCopyInfo->sType != VK_STRUCTURE_TYPE_TENSOR_COPY_INFO_KHR) return VK_FALSE;`

### Sub-phase 28b: Graph validation (6 functions)

- [X] T158 [P] In `layers/validation/graph_validation.c`, add sType checks to 6 functions:
  - `vk_ml_validate_graph_create`: after NULL guard, add `if (pCreateInfo->sType != VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR) return VK_FALSE;`
  - `vk_ml_validate_convolution_desc`: after NULL guard, add `if (desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR) return VK_FALSE;`
  - `vk_ml_validate_gemm_desc`: after NULL guard, add `if (desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR) return VK_FALSE;`
  - `vk_ml_validate_pooling_desc`: after NULL guard, add `if (desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR) return VK_FALSE;`
  - `vk_ml_validate_normalization_desc`: after NULL guard, add `if (desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR) return VK_FALSE;`
  - `vk_ml_validate_elementwise_desc`: after NULL guard, add `if (desc->sType != VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR) return VK_FALSE;`

### Sub-phase 28c: Session validation (1 function)

- [X] T159 [P] In `layers/validation/session_validation.c`, add sType check to `vk_ml_validate_session_create`: after NULL guard, add `if (pCreateInfo->sType != VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR) return VK_FALSE;`

### Sub-phase 28d: Dispatch validation (1 function)

- [X] T160 [P] In `layers/validation/dispatch_validation.c`, add sType check to `vk_ml_validate_dispatch`: after NULL guard, add `if (pDispatchInfo->sType != VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR) return VK_FALSE;`

### Sub-phase 28e: Barrier validation (2 functions)

- [X] T161 [P] In `layers/validation/barrier_validation.c`, add sType checks to 2 functions:
  - `vk_ml_validate_tensor_memory_barrier`: after NULL guard, add `if (barrier->sType != VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR) return VK_FALSE;`
  - `vk_ml_validate_tensor_dependency_info`: after NULL guard, add `if (depInfo->sType != VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR) return VK_FALSE;`

### Sub-phase 28f: Build + test verification

- [X] T162 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass. All existing tests use correct sType values and should not be affected.

**Checkpoint**: All 14 validation functions now check `sType` as their first validation step (after NULL). Wrong `sType` is rejected with `VK_FALSE`. All 13 tests pass.

---

### Phase 28 Dependencies

```text
Sub-phase 28a (tensor):   T157 — tensor_validation.c
Sub-phase 28b (graph):    T158 — [P] graph_validation.c
Sub-phase 28c (session):  T159 — [P] session_validation.c
Sub-phase 28d (dispatch): T160 — [P] dispatch_validation.c
Sub-phase 28e (barrier):  T161 — [P] barrier_validation.c
Sub-phase 28f (verify):   T162 — depends on T157-T161

Total: 6 tasks. T157-T161 all parallelizable (different files). T162 final.
```

---

## Phase 29: Review Remediation — M8 + M9 (Missing Handle Validation in Create Functions)

**Goal**: Add null-handle checks for `pCreateInfo->tensor` in `vkCreateTensorViewKHR` (M8) and `pCreateInfo->graph` in `vkCreateMLSessionKHR` (M9). Both are ICD-level defensive guards returning `VK_ERROR_UNKNOWN`, consistent with M1/M2/M3 precedent. Resolves MEDIUM findings M8 and M9 from `review-findings.md`.

### Sub-phase 29a: Add handle guards (parallelizable)

- [X] T163 [P] In `src/tensor_view.c`, after the existing NULL guard `if (!pCreateInfo || !pView) return VK_ERROR_UNKNOWN;` (line 19-20), add `if (pCreateInfo->tensor == VK_NULL_HANDLE) return VK_ERROR_UNKNOWN;` before the allocation on line 22.

- [X] T164 [P] In `src/ml_session.c`, after the existing NULL guard `if (!pCreateInfo || !pSession) return VK_ERROR_UNKNOWN;` (line 19-20), add `if (pCreateInfo->graph == VK_NULL_HANDLE) return VK_ERROR_UNKNOWN;` before the allocation on line 22.

### Sub-phase 29b: Build + test verification

- [X] T165 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: Both create functions now reject null parent handles at the ICD level. All 13 tests pass.

---

### Phase 29 Dependencies

```text
Sub-phase 29a (guards):  T163, T164 — [P] (different files)
Sub-phase 29b (verify):  T165 — depends on T163, T164

Total: 3 tasks. T163, T164 parallelizable. T165 final.
```

---

## Phase 30: Review Remediation — M10 (Convolution kernel dimension validation)

**Goal**: Reject `kernelWidth == 0` or `kernelHeight == 0` in the convolution descriptor validation. A 0x0 kernel is meaningless. Validation-layer-only change. Resolves MEDIUM finding M10 from `review-findings.md`.

- [X] T166 In `layers/validation/graph_validation.c`, inside `vk_ml_validate_convolution_desc`, add a kernel-dimension check after the sType check (line 88) and before the stride check (line 90): `/* VUID_CONV_KERNEL */ if (desc->kernelWidth == 0 || desc->kernelHeight == 0) return VK_FALSE;`

- [X] T167 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: Convolution validation now rejects zero-dimension kernels. All 13 tests pass.

---

### Phase 30 Dependencies

```text
T166 — sequential (single file)
T167 — depends on T166

Total: 2 tasks.
```

---

## Phase 31: Review Remediation — M11 (Convolution groupCount = 0 not rejected)

**Goal**: Reject `groupCount == 0` in convolution descriptor validation. Zero is unconditionally invalid regardless of tensor shape. Validation-layer-only change. Resolves MEDIUM finding M11 from `review-findings.md`.

- [X] T168 In `layers/validation/graph_validation.c`, inside `vk_ml_validate_convolution_desc`, add `/* VUID_CONV_GROUP_COUNT */ if (desc->groupCount == 0) return VK_FALSE;` immediately before the existing stub comment at line 114.

- [X] T169 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: Convolution validation now rejects zero groupCount. All 13 tests pass.

---

### Phase 31 Dependencies

```text
T168 — sequential (single file)
T169 — depends on T168

Total: 2 tasks.
```

---

## Phase 32: Review Remediation — M12 (Tensor usage flags never validated)

**Goal**: Reject zero usage flags and undefined usage bits in `vk_ml_validate_tensor_create`. Validation-layer-only change. Resolves MEDIUM finding M12 from `review-findings.md`.

- [X] T170 In `layers/validation/tensor_validation.c`, inside `vk_ml_validate_tensor_create`, after the stride alignment check (line 65) and before `return VK_TRUE` (line 67), add: (1) `/* VUID_TENSOR_USAGE */ if (desc->usage == 0) return VK_FALSE;` to reject zero usage, and (2) `const VkFlags validUsageMask = 0x7F; if (desc->usage & ~validUsageMask) return VK_FALSE;` to reject undefined bits (bits 0-6 are the 7 defined `VK_TENSOR_USAGE_*_BIT_KHR` values).

- [X] T171 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: Tensor creation validation now rejects zero and invalid usage flags. All 13 tests pass.

---

### Phase 32 Dependencies

```text
T170 — sequential (single file)
T171 — depends on T170

Total: 2 tasks.
```

---

## Phase 33: Review Remediation — M13 (Tensor view doesn't check tensor has memory bound)

**Goal**: Reject tensor view creation when the source tensor has no memory bound. Validation-layer-only change. Resolves MEDIUM finding M13 from `review-findings.md`.

- [X] T172 In `layers/validation/tensor_validation.c`, inside `vk_ml_validate_tensor_view_create`, after the sType check (line 84) and before the format validation (line 86), add: `/* VUID_TENSOR_VIEW_MEMORY_BOUND */ if (!tensor->memoryBound) return VK_FALSE;`

- [X] T173 Build with `cmake --build build` — zero warnings. Run `ctest --output-on-failure` — all 13 tests pass.

**Checkpoint**: Tensor view validation now rejects views over tensors without bound memory. All 13 tests pass.

---

### Phase 33 Dependencies

```text
T172 — sequential (single file)
T173 — depends on T172

Total: 2 tasks.
```

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Tests MUST fail before implementing (Constitution Principle IV)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All tasks reference exact file paths from plan.md
- VUID strings in validation layer MUST match spec/VK_KHR_ml_primitives.adoc exactly
