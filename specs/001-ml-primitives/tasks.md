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

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Tests MUST fail before implementing (Constitution Principle IV)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All tasks reference exact file paths from plan.md
- VUID strings in validation layer MUST match spec/VK_KHR_ml_primitives.adoc exactly
