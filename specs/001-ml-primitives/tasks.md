# Tasks: v1.0 Release Readiness

**Input**: Design documents from `/specs/001-ml-primitives/`
**Prerequisites**: plan.md, spec.md, v1.0-readiness.plan.md, research.md, data-model.md, contracts/

**Tests**: Test tasks are included — the spec mandates validation layer checks with corresponding test cases for all VUIDs (Constitution IV, SC-007).

**Organization**: Tasks follow the 6-phase dependency order from the readiness plan. User story labels map fixes to the user story they harden. Infrastructure/polish tasks are cross-cutting.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task hardens (US1=Tensors, US2=Graphs, US3=Sessions/Dispatch)
- Include exact file paths in descriptions

## Path Conventions

- Public header: `include/vulkan/vulkan_ml_primitives.h`
- ICD source: `src/*.c`, `src/internal.h`
- Validation layer: `layers/validation/*.c`, `layers/validation/vk_ml_validation.h`
- Tests: `tests/cts/*.c`, `tests/unit/*.c`, `tests/validation/*.c`
- Build: `CMakeLists.txt`, `cmake/*.in`, `.github/workflows/ci.yml`

---

## Phase 1: Foundational — Public API Surface Fixes (CRITICAL)

**Purpose**: Fix the public header so all downstream code (ICD, validation, tests) has correct sType values and naming conventions. BLOCKS all subsequent phases.

**⚠️ CRITICAL**: No ICD, validation, or test work can begin until this phase is complete. sType collisions cause silent misinterpretation of structures.

- [x] T001 Reassign colliding VkStructureType values (1000559020–23 → 1000559024–27) for ML_GRAPH_DISPATCH_INFO_KHR, ML_TENSOR_BINDING_KHR, TENSOR_DEPENDENCY_INFO_KHR, TENSOR_COPY_KHR in include/vulkan/vulkan_ml_primitives.h
- [x] T002 Rename VK_FORMAT_R8_BOOL to VK_FORMAT_R8_BOOL_KHR in include/vulkan/vulkan_ml_primitives.h and update all references in src/internal.h, src/feature_query.c, and test files
- [x] T003 Move VkMLResizeModeKHR enum definition to the Enumerations (T005) section in include/vulkan/vulkan_ml_primitives.h (after VkMLTensorBindingTypeKHR, before Bitmask types)

**Checkpoint**: `cmake --build build` compiles cleanly. All existing tests pass with the new sType values and renamed format enum.

---

## Phase 2: User Story 1 — Tensor ICD Hardening (Priority: P1)

**Goal**: Fix all ICD correctness issues in tensor creation, binding, view, and copy operations so that tensor resource management is robust against invalid input, NULL pointers, and overflow.

**Independent Test**: Create tensors, bind memory, create views, copy tensors — all paths handle edge cases (NULL handles, wrong sType, missing extents) without crashing or silent corruption.

### Implementation for User Story 1

- [x] T004 [P] [US1] Add sType validation check after NULL checks in vkCreateTensorKHR in src/tensor.c
- [x] T005 [P] [US1] Add sType validation check after NULL checks in vkCreateTensorViewKHR in src/tensor_view.c
- [x] T006 [P] [US1] Replace `continue` with `return VK_ERROR_UNKNOWN` for NULL tensor handle in vkBindTensorMemoryKHR in src/tensor.c
- [x] T007 [P] [US1] Add pExtents NULL check to copy region validation loop in src/tensor_copy.c
- [x] T008 [P] [US1] Replace `(int)` casts with `(uint32_t)` for sType comparisons in src/tensor_copy.c

**Checkpoint**: Tensor lifecycle ICD functions reject invalid sType, NULL handles, and missing extents. Build and existing tests pass.

---

## Phase 3: User Story 2 — Graph ICD Hardening (Priority: P2)

**Goal**: Fix graph creation deep-copy failures for new descriptor types (concat, reshape, transpose, resize) and guard against NULL dereferences in scratch size calculation.

**Independent Test**: Create graphs using all 10 descriptor types including concat/reshape/transpose/resize. Verify graph creation succeeds and scratch size is computed without crashes even when description arrays are NULL.

### Implementation for User Story 2

- [x] T009 [US2] Add concat, reshape, transpose, resize cases to op_desc_size_by_stype() in src/ml_graph.c returning sizeof for each descriptor struct
- [x] T010 [US2] Add post-copy deep-copy logic in deep_copy_op_desc() for VkMLPrimitiveDescReshapeKHR.pOutputDimensions and VkMLPrimitiveDescTransposeKHR.pPermutation in src/ml_graph.c
- [x] T011 [US2] Add corresponding free logic in free_node_deep_data() for reshape pOutputDimensions and transpose pPermutation pointer members in src/ml_graph.c
- [x] T012 [US2] Guard scratch size calculation loops with NULL checks on externalInputDescs, externalOutputDescs, constantWeightDescs in src/ml_graph.c

**Checkpoint**: Graph creation with concat/reshape/transpose/resize descriptors returns VK_SUCCESS. Scratch calculation does not dereference NULL. Build and existing tests pass.

---

## Phase 4: User Story 3 — Session and Dispatch ICD Hardening (Priority: P3)

**Goal**: Fix session struct initialization, enforce scratch size validation, and harden dispatch parameter checks so that session creation and graph dispatch are robust.

**Independent Test**: Create sessions with explicit scratch memory that is too small (expect error), create sessions with no scratch and auto-alloc disabled (expect error), dispatch with valid and invalid parameters.

### Implementation for User Story 3

- [x] T013 [P] [US3] Add memset(session, 0, sizeof(VkMLSessionKHR_T)) after allocation in vkCreateMLSessionKHR in src/ml_session.c
- [x] T014 [P] [US3] Add scratch memory size validation (pCreateInfo->scratchMemorySize < g->scratchMemorySize → return VK_ERROR_UNKNOWN) in src/ml_session.c
- [x] T015 [P] [US3] Replace `(int)` cast with `(uint32_t)` for sType comparison in vkCmdDispatchMLGraphKHR in src/ml_dispatch.c

**Checkpoint**: Sessions reject insufficient scratch. Uninitialized memory bugs eliminated. Build and existing tests pass (may need to update test_oom.c if scratch size validation changes behavior).

---

## Phase 5: Validation Layer Hardening (CRITICAL + HIGH)

**Goal**: Fix all validation layer bugs — integer overflow in tensor view range check, stack buffer overflow in DFS, missing NULL checks, activation descriptor validation, and new descriptor type handling.

**Independent Test**: Validation functions correctly reject UINT32_MAX overflow in view ranges, graphs with >256 nodes, NULL pExtents in copy regions, invalid activation descriptors, and unknown sTypes.

### Implementation for Tensor Validation (US1)

- [x] T016 [P] [US1] Rewrite tensor view range check in vk_ml_validate_tensor_view_create as overflow-safe subtraction in layers/validation/tensor_validation.c
- [x] T017 [P] [US1] Add pExtents NULL check for copy regions in vk_ml_validate_tensor_copy in layers/validation/tensor_validation.c
- [x] T018 [P] [US1] Add srcTensor/dstTensor VK_NULL_HANDLE checks before same-tensor check in vk_ml_validate_tensor_copy in layers/validation/tensor_validation.c

### Implementation for Graph Validation (US2)

- [x] T019 [US2] Add nodeCount > VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT guard before DFS stack allocation in vk_ml_validate_graph_create in layers/validation/graph_validation.c
- [x] T020 [US2] Declare vk_ml_validate_activation_desc() in layers/validation/vk_ml_validation.h
- [x] T021 [US2] Implement vk_ml_validate_activation_desc() in layers/validation/graph_validation.c: validate sType, activationType enum range (0–5), param0/param1 finiteness, clamp param0 <= param1
- [x] T022 [US2] Wire vk_ml_validate_activation_desc() into the per-node descriptor switch for VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR case in layers/validation/graph_validation.c
- [x] T023 [US2] Change default case in per-node descriptor switch from `break` to `return VK_FALSE` in layers/validation/graph_validation.c
- [x] T024 [US2] Add cases for concat, reshape, transpose, resize sTypes in per-node descriptor switch, delegating to vk_ml_validate_primitive_desc() in layers/validation/graph_validation.c

### Implementation for sType Cast Cleanup (Cross-cutting)

- [x] T025 Replace all remaining `(int)` casts on VkStructureType comparisons with `(uint32_t)` in layers/validation/tensor_validation.c, layers/validation/graph_validation.c, layers/validation/session_validation.c, layers/validation/dispatch_validation.c

**Checkpoint**: All validation functions handle edge cases. Overflow, stack overflow, NULL pointer, and unknown sType attacks are caught. Build and tests pass.

---

## Phase 6: Infrastructure and Build Fixes

**Purpose**: Fix build system, CI pipeline, packaging, and documentation issues that affect portability and developer experience.

- [x] T026 [P] Fix clang-tidy configuration: replace explicit -checks with --config-file in CMAKE_C_CLANG_TIDY in CMakeLists.txt
- [x] T027 [P] Fix pkg-config template: replace hardcoded libdir/includedir with @CMAKE_INSTALL_FULL_LIBDIR@/@CMAKE_INSTALL_FULL_INCLUDEDIR@ in cmake/vk_ml_primitives.pc.in
- [x] T028 [P] Add NAMESPACE VulkanML:: to install(EXPORT) in CMakeLists.txt
- [x] T029 [P] Add docs/html/, docs/latex/ entries to .gitignore
- [x] T030 [P] Replace hardcoded lunarg-vulkan-noble.list with dynamic $(lsb_release -cs) in both build and static-analysis jobs in .github/workflows/ci.yml
- [x] T031 [P] Add clang-format --dry-run --Werror check step to CI in .github/workflows/ci.yml
- [x] T032 [P] Update compiler version requirements to GCC 12+, Clang 15+ in README.md
- [x] T033 [P] Update compiler version requirements to GCC 12+, Clang 15+, MSVC 2022+ in CONTRIBUTING.md

**Checkpoint**: `cmake --build build` succeeds. CI configuration is portable. Packaging produces correct paths on lib64 systems. `.gitignore` is comprehensive.

---

## Phase 7: Test Coverage

**Purpose**: Add tests that exercise every fix from Phases 2–5, ensuring regressions are caught.

### Tests for Tensor Fixes (US1)

- [x] T034 [P] [US1] Add test_tensor_view_overflow_bounds in tests/unit/test_descriptor_validation.c: overflow-safe bounds check
- [x] T035 [P] [US1] Add test_bind_null_tensor_handle in tests/cts/test_tensor_lifecycle.c: NULL handle returns VK_ERROR_UNKNOWN
- [x] T036 [P] [US1] Add test_activation_valid_relu, test_activation_invalid_type, test_activation_clamp_param0_gt_param1, test_activation_clamp_valid in tests/unit/test_descriptor_validation.c

### Tests for Graph Fixes (US2)

- [x] T037 [P] [US2] Add test_diamond_dag_valid and test_three_node_cycle in tests/unit/test_dag_validation.c
- [x] T038 [P] [US2] Add test_concat_with_descriptor, test_reshape_with_descriptor, test_transpose_with_descriptor, test_resize_with_descriptor in tests/cts/test_ml_graph.c
- [x] T039 [P] [US3] Add test_scratch_size_zero_with_memory and test_scratch_offset_unaligned in tests/cts/test_ml_session.c
- [x] T040 [P] [US1] Add test_create_tensor_wrong_stype in tests/cts/test_tensor_lifecycle.c
- [x] T041 [P] [US1] Add test_copy_pextents_null in tests/unit/test_descriptor_validation.c
- [x] T042 [P] [US2] Add test_unknown_operation_stype in tests/unit/test_dag_validation.c

### Tests for Session Fixes (US3)

- [x] T043 [P] [US2] Add test_reshape_transpose_deep_copy in tests/cts/test_ml_graph.c: deep-copy verification

**Checkpoint**: All new tests pass. `ctest --test-dir build --output-on-failure` shows zero failures.

---

## Phase 8: Release Polish

**Purpose**: Version bump, changelog finalization, and documentation update for the 1.0.0 tag.

- [x] T044 Update CHANGELOG.md: move [Unreleased] content to [1.0.0] section, add all v1.0 fixes from this plan
- [x] T045 [P] Bump VERSION from 0.1.0 to 1.0.0 in CMakeLists.txt and Doxyfile
- [x] T046 Update README.md: verify test counts, version references, and all sections reflect final 1.0.0 state
- [x] T047 Run quickstart.md validation: execute full build-test-analyze cycle

**Checkpoint**: Repository is ready for `git tag -a v1.0.0`. All tests pass, static analysis clean, docs accurate.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (API Surface)**: No dependencies — start immediately. **BLOCKS all other phases**.
- **Phase 2 (US1 ICD)**: Depends on Phase 1 completion
- **Phase 3 (US2 ICD)**: Depends on Phase 1 completion
- **Phase 4 (US3 ICD)**: Depends on Phase 1 completion
- **Phase 5 (Validation)**: Depends on Phase 1 completion
- **Phase 6 (Infrastructure)**: Depends on Phase 1 completion (sType values in clang-tidy checks)
- **Phase 7 (Tests)**: Depends on Phases 2–5 completion (tests exercise the fixes)
- **Phase 8 (Polish)**: Depends on all previous phases

### User Story Dependencies

- **US1 (Tensors, P1)**: Phase 1 → Phase 2 (T004–T008) + Phase 5 (T016–T018) → Phase 7 (T034–T036)
- **US2 (Graphs, P2)**: Phase 1 → Phase 3 (T009–T012) + Phase 5 (T019–T024) → Phase 7 (T037–T042)
- **US3 (Sessions, P3)**: Phase 1 → Phase 4 (T013–T015) + Phase 5 (T025) → Phase 7 (T043)
- **US4 (Sync, P4)**: No fixes needed — existing implementation is correct
- **US5 (SPIR-V, P5)**: No fixes needed — existing implementation is correct

### Within Each Phase

- Phase 1: T001 first (sType collision), then T002 and T003 in parallel
- Phase 2: T004–T008 can all run in parallel (different files)
- Phase 3: T009 before T010–T011 (size function before deep-copy logic); T012 independent
- Phase 4: T013–T015 can all run in parallel (different files)
- Phase 5: T016–T018 parallel; T019 independent; T020 before T021 before T022; T023–T024 independent
- Phase 6: All T026–T033 can run in parallel (all different files)
- Phase 7: All T034–T043 can run in parallel (different test files)
- Phase 8: T044 first (changelog), then T045–T046 parallel, then T047 last

### Parallel Opportunities

- After Phase 1 completes, **Phases 2–6 can all proceed in parallel** — they touch different files with no cross-dependencies
- Within Phase 2: All 5 tasks target different source files
- Within Phase 5: All tensor validation tasks (T016–T018) target the same file but different functions — serialize within file
- Within Phase 6: All 8 tasks target different files — full parallelism
- Within Phase 7: All 10 test tasks can run in parallel (different test files or additive functions)

---

## Parallel Example: Phase 2 + 3 + 4 (after Phase 1)

```
# After Phase 1 is complete, launch all ICD fixes in parallel:
Task: "T004 [P] [US1] Add sType validation in src/tensor.c"
Task: "T005 [P] [US1] Add sType validation in src/tensor_view.c"
Task: "T006 [P] [US1] Replace continue with error in src/tensor.c"
Task: "T007 [P] [US1] Add pExtents NULL check in src/tensor_copy.c"
Task: "T008 [P] [US1] Replace (int) casts in src/tensor_copy.c"
Task: "T009 [US2] Add new desc cases in src/ml_graph.c"
Task: "T013 [P] [US3] Zero-init session in src/ml_session.c"
Task: "T014 [P] [US3] Validate scratch size in src/ml_session.c"
Task: "T015 [P] [US3] Replace (int) cast in src/ml_dispatch.c"
```

---

## Parallel Example: Phase 7 (all tests)

```
# After all fixes are in, launch all tests in parallel:
Task: "T034 [P] [US1] test_tensor_view_uint32_overflow in tests/unit/test_validation_coverage.c"
Task: "T035 [P] [US1] test_tensor_copy_null_extents in tests/unit/test_validation_coverage.c"
Task: "T036 [P] [US1] test_tensor_copy_null_handles in tests/unit/test_validation_coverage.c"
Task: "T037 [P] [US2] test_valid_diamond_dag in tests/unit/test_dag_validation.c"
Task: "T038 [P] [US2] Activation descriptor tests in tests/unit/test_descriptor_validation.c"
Task: "T039-T042 [P] [US2] New op type graph tests in tests/cts/test_ml_graph.c"
Task: "T043 [P] [US3] test_session_scratch_too_small in tests/cts/test_ml_session.c"
```

---

## Implementation Strategy

### MVP First (Phase 1 + Critical Fixes Only)

1. Complete Phase 1: API Surface Fixes (T001–T003) — eliminates sType collisions
2. Complete Critical ICD fixes: T009–T012 (graph deep-copy + NULL guard)
3. Complete Critical validation fixes: T016, T019 (overflow + stack overflow)
4. **STOP and VALIDATE**: Build + all existing tests pass
5. This resolves all 5 CRITICAL issues — the minimum for a defensible 1.0

### Incremental Delivery

1. Phase 1 → API frozen ✓
2. Phase 2–4 → ICD hardened across all user stories ✓
3. Phase 5 → Validation layer hardened ✓
4. Phase 6 → Infrastructure portable ✓
5. Phase 7 → Full regression test coverage ✓
6. Phase 8 → Release tagged `v1.0.0` ✓

### Parallel Strategy

With multiple work streams:

1. Everyone completes Phase 1 together (small, 3 tasks)
2. Once Phase 1 is done:
   - Stream A: Phases 2 + 4 (tensor + session ICD fixes) — 8 tasks
   - Stream B: Phase 3 (graph ICD fixes) — 4 tasks
   - Stream C: Phase 5 (validation layer) — 10 tasks
   - Stream D: Phase 6 (infrastructure) — 8 tasks
3. All streams join for Phase 7 (tests) and Phase 8 (polish)

---

## Summary

| Phase | Tasks | User Stories | Parallel? |
|-------|-------|-------------|-----------|
| 1. API Surface | 3 (T001–T003) | All | T002, T003 after T001 |
| 2. US1 ICD | 5 (T004–T008) | US1 | All parallel |
| 3. US2 ICD | 4 (T009–T012) | US2 | T009 → T010–T011; T012 parallel |
| 4. US3 ICD | 3 (T013–T015) | US3 | All parallel |
| 5. Validation | 10 (T016–T025) | US1, US2, US3 | Mostly parallel |
| 6. Infrastructure | 8 (T026–T033) | Cross-cutting | All parallel |
| 7. Tests | 10 (T034–T043) | US1, US2, US3 | All parallel |
| 8. Polish | 4 (T044–T047) | Release | Sequential |
| **Total** | **47** | | |

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks in same phase
- [Story] label maps task to the user story it hardens for traceability
- US4 (Synchronization) and US5 (SPIR-V) have no fixes — their implementations passed review
- Tests in Phase 7 are REQUIRED per Constitution IV (test-first with validation layers)
- Commit after each phase or logical group
- Stop at any checkpoint to validate independently
- The 47 tasks cover all 35 issues from the readiness review plus 12 test tasks for regression coverage
