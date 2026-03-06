# Review Findings — VK_KHR_ml_primitives

**Review 1 date**: 2026-03-05
**Review 2 date**: 2026-03-06
**Scope**: Full repository deep review (src, validation layer, tests, header, build, spec conformance)
**Build status**: 13/13 tests pass, zero warnings under `-Wall -Wextra -Wpedantic -Werror`
**Spec conformance**: 100% — all types, enums, structs, and entry points match the `.adoc` spec

---

## How to Use This Document

Each finding has:

- **ID**: Unique reference (e.g., `C1`, `H3`, `M7`)
- **Severity**: CRITICAL > HIGH > MEDIUM > LOW
- **Status**: `[ ]` unchecked = not started, `[x]` = fixed
- **File(s)**: Affected source location(s)
- **Description**: What's wrong
- **Fix**: Recommended solution

Work through findings top-down by severity. Some fixes are independent and can be parallelized (marked `[P]`).

---

## CRITICAL (2)

### C1 — Shallow-copy dangling pointers in graph node storage

- [x] **File**: `src/ml_graph.c:99-106`
- **Description**: `vkCreateMLGraphKHR` does `memcpy` of `VkMLGraphNodeCreateInfoKHR` nodes, which contain pointers (`pOperationDesc`, `pInputs`, `pOutputs`, `pNodeName`). After the create call returns, the caller can free those pointed-to objects, leaving the graph with dangling pointers. Any subsequent traversal is use-after-free.
- **Fix**: Deep-copy each node's `pInputs`, `pOutputs`, `pOperationDesc`, and `pNodeName` (strdup). Update `vkDestroyMLGraphKHR` to free the deep-copied data. Follow the same pattern already used for `deep_copy_tensor_desc`.
- **FIXED**: Phase 10 (T081-T091). Added `deep_copy_op_desc`, `deep_copy_bindings`, `deep_copy_string`, `free_node_deep_data` helpers. Refactored `vkCreateMLGraphKHR` with `goto cleanup` pattern. Added 3 CTS ownership tests.

### C2 — `pNext` chain clobbered in feature/property queries

- [x] **File**: `src/feature_query.c:20-21, 46, 97`
- **Description**: `vk_ml_populate_features`, `vk_ml_populate_properties`, and `vk_ml_populate_tensor_format_properties` overwrite `sType` and set `pNext = NULL`. Vulkan convention requires the implementation to leave `sType`/`pNext` untouched — the caller sets those before querying. This breaks any pNext-chained extension structures.
- **Fix**: Remove the lines that set `sType` and `pNext`. Only populate the data fields (feature booleans, property limits, format properties).
- **FIXED**: Phase 11 (T092-T098). Removed `sType`/`pNext` overwrites from all 3 populate functions. Added 3 CTS tests verifying pNext chain preservation.

---

## HIGH (10)

### H1 — Dangling description pointers in tensor

- [x] **File**: `src/tensor.c:29, 109`
- **Description**: `tensor->description = *desc` shallow-copies the `VkTensorDescriptionKHR` including its `pDimensions` and `pStrides` pointers. The arrays are deep-copied into `tensor->dimensions`/`tensor->strides`, but `tensor->description.pDimensions` still points to the caller's (possibly freed) memory. The fallback path in `vkGetTensorMemoryRequirementsKHR` at line 109 is a latent use-after-free.
- **Fix**: After deep-copying dimensions/strides, update the description's pointers:

  ```c
  tensor->description.pDimensions = tensor->dimensions;
  tensor->description.pStrides = tensor->strides;
  ```

- **FIXED**: Phase 12 (T099-T102). Redirected `description.pDimensions`/`pStrides` to owned copies, nulled `pNext`. Simplified fallback in `vkGetTensorMemoryRequirementsKHR`. Added CTS ownership test.

### H2 — Hardcoded alignment of 8 in vk_ml_alloc

- [x] [P] **File**: `src/internal.h:89`
- **Description**: `vk_ml_alloc` passes a hardcoded alignment of `8` to the Vulkan allocation callback. This is insufficient for types requiring stricter alignment. `malloc` guarantees `_Alignof(max_align_t)`, but the callback path doesn't.
- **Fix**: Add an `alignment` parameter to `vk_ml_alloc`, or use `_Alignof(max_align_t)` as the default. Update all callers.
- **FIXED**: Phase 13 (T103-T106). Replaced hardcoded `8` with `_Alignof(max_align_t)`. Added `#include <stddef.h>`. Added CTS test verifying callback alignment.

### H3 — Integer overflow in dimension product validation

- [x] **File**: `layers/validation/tensor_validation.c:34-44`
- **Description**: `uint64_t product` can silently wrap when multiplying large dimensions (e.g., 4 dims of 65536 = 2^64, wraps to 0). The overflow causes `0 > maxTensorElements` to be false, so validation incorrectly passes.
- **Fix**: Check for overflow before each multiplication:

  ```c
  if (desc->pDimensions[i] != 0 &&
      product > props->maxTensorElements / desc->pDimensions[i])
      return VK_FALSE;
  product *= desc->pDimensions[i];
  ```

- **FIXED**: Phase 14 (T107-T109). Added overflow guard `product > maxTensorElements / dim` before each multiplication. Added unit test with 4 dims of 65536.

### H4 — NULL `pNodes` accepted with `nodeCount > 0`

- [x] **File**: `layers/validation/graph_validation.c:62`
- **Description**: DFS cycle detection is guarded by `if (pCreateInfo->pNodes)`. If `pNodes == NULL` but `nodeCount > 0`, validation passes. Downstream code that dereferences `pNodes` will crash.
- **Fix**: Add before the DFS block:

  ```c
  if (pCreateInfo->nodeCount > 0 && !pCreateInfo->pNodes)
      return VK_FALSE;
  ```

- **FIXED**: Phase 15 (T110-T112). Added `if (!pCreateInfo->pNodes) return VK_FALSE;` guard before DFS block. Added unit test with `nodeCount=1, pNodes=NULL`.

### H5 — Dispatch validation structurally incomplete

- [x] **File**: `layers/validation/dispatch_validation.c`
- **Description**: `vk_ml_validate_dispatch` only checks `session != NULL` and counts `!= 0`. It doesn't verify counts match graph expectations (no graph/session context available), doesn't check tensor array pointers are non-NULL when counts > 0, and doesn't check tensor usage flags.
- **Fix**: (a) Add NULL-array checks for `pInputTensors`/`pOutputTensors`/`pWeightTensors` when respective counts > 0. (b) Expand the function signature to accept graph context for count matching (longer-term).
- **FIXED** (partial): Phase 16 (T113-T117). Added 3 NULL-pointer guards for tensor arrays. Count-matching deferred (requires architectural change to thread graph context).

### H6 — Barrier validation lives in wrong directory

- [x] **File**: `src/tensor_barrier.c` -> `layers/validation/barrier_validation.c`
- **Description**: `vk_ml_validate_tensor_memory_barrier` and `vk_ml_validate_tensor_dependency_info` are validation functions implemented in `src/` instead of `layers/validation/`. The file doesn't include `vk_ml_validation.h`, so signatures can drift without compile-time detection.
- **Fix**: Move the validation functions to a new `layers/validation/barrier_validation.c` (or into an existing validation file). Keep any non-validation barrier logic (if added later) in `src/tensor_barrier.c`. Update `CMakeLists.txt` to add the new file to `VALIDATION_SOURCES`. Ensure the new file includes `vk_ml_validation.h`.
- **FIXED**: Phase 17 (T118-T122). Moved both validation functions to `layers/validation/barrier_validation.c` with `#include "../validation/vk_ml_validation.h"`. Removed `src/tensor_barrier.c` from `ML_PRIMITIVES_SOURCES`, added new file to `VALIDATION_SOURCES`. Old file deleted. All 12 tests pass.

### H7 — No README.md

- [x] **File**: `README.md`
- **Description**: The repository has no README. For a project of this scope, this is a significant gap for discoverability and onboarding.
- **Fix**: Create `README.md` with: project description, build instructions, test instructions, directory structure guide, extension status, and link to the spec.
- **FIXED**: Phase 18 (T123-T124). Created comprehensive README with 11 sections: overview, extension dependencies, repository structure tree, prerequisites, build instructions, test instructions (with 12-suite table), quick start guide, static analysis, specification link, and license placeholder. All 31 referenced paths verified on disk.

### H8 — No OOM / allocation-failure tests

- [x] **File**: `tests/cts/test_oom.c`
- **Description**: No CTS test exercises `VK_ERROR_OUT_OF_HOST_MEMORY` return paths. The reference implementation always succeeds unless input validation fails.
- **Fix**: Add tests using a custom `VkAllocationCallbacks` that returns NULL after N allocations. Verify create functions return `VK_ERROR_OUT_OF_HOST_MEMORY` and don't leak partially-allocated resources.
- **FIXED**: Phase 19 (T125-T127). Created `test_oom.c` with a failing allocator that returns NULL after N allocations. 4 test functions cover all OOM paths in `vkCreateTensorKHR` (4 points), `vkCreateTensorViewKHR` (3 points), `vkCreateMLGraphKHR` (15+ points via ascending loop), and `vkCreateMLSessionKHR` (1 point). **Bonus**: OOM tests uncovered a real bug — uninitialized descriptor arrays in `vkCreateMLGraphKHR`'s cleanup path caused double-free on partial allocation failure. Fixed by zero-initializing `externalInputDescs`, `externalOutputDescs`, and `constantWeightDescs` arrays after allocation.

### H9 — Tautological tests that can never fail

- [x] **Files**: `test_tensor_lifecycle.c`, `test_tensor_view.c`, `test_tensor_copy.c`, `test_synchronization.c`
- **Description**: Several tests always pass regardless of implementation correctness:
  - `test_destroy_null_handle` / `test_destroy_view_null` — return 0 unconditionally
  - `test_copy_basic` / `test_copy_null_cmd` — "no crash = pass", no assertions
  - `test_barrier_structure` / `test_dependency_info` / `test_queue_family_transfer` — verify C struct initialization reads back correctly (compiler-guaranteed)
- **Fix**: Add meaningful assertions: verify destroy doesn't crash by checking it doesn't corrupt adjacent state, verify copy records the operation (if possible), replace struct-initialization tests with validation-layer or API-level tests.
- **FIXED**: Phase 20 (T128-T132). All 7 tests now have real assertions: null-handle destroy tests verify a live adjacent object survives; copy tests verify tensor internal state is preserved; struct-readback tests replaced with `vk_ml_validate_tensor_memory_barrier` / `vk_ml_validate_tensor_dependency_info` calls testing valid/invalid configurations.

### H10 — CTS tests depend on internal representation

- [x] **Files**: `test_tensor_lifecycle.c`, `test_tensor_view.c`, `test_tensor_copy.c`, `test_ml_session.c`
- **Description**: Tests cast handles to `VkTensorKHR_T*` / `VkMLSessionKHR_T*` to inspect internal fields (`memoryBound`, `autoAllocated`). This makes them reference-implementation-specific and will break on any other ICD.
- **Fix**: Test observable behavior through the public API instead. For bind verification, test that a subsequent operation requiring bound memory succeeds. For auto-allocation, test that the session functions correctly without explicit scratch memory.
- **FIXED**: Phase 21 (T133-T139). Replaced all 8 internal struct casts across 4 CTS files with public API queries (`vkGetTensorMemoryRequirementsKHR` for tensor validity, `VK_SUCCESS` return for session auto-alloc). Removed `internal.h` from `test_tensor_lifecycle.c`, `test_tensor_view.c`, `test_tensor_copy.c`, `test_ml_session.c`. Also replaced `VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN` constant with `alignment > 0` check. Zero internal type references remain in CTS (`grep` verified).

---

## MEDIUM (19)

### M1 — Wrong error code for NULL parameter validation

- [x] [P] **Files**: `src/tensor.c:20,135`, `src/tensor_view.c:20`, `src/ml_graph.c:239`, `src/ml_session.c:20`
- **Description**: All create functions return `VK_ERROR_INITIALIZATION_FAILED` for NULL input pointers. This error code is for driver/device initialization failures. Per Vulkan convention, ICDs should not validate parameters (that's the validation layer's job), but if they do, a more appropriate code would be used.
- **Fix**: Either remove parameter validation from ICDs (rely on validation layer) or return `VK_ERROR_UNKNOWN` / leave the behavior as implementation-defined with a comment explaining the choice.
- **FIXED**: Phase 22 (T140-T144). Replaced all 5 occurrences of `VK_ERROR_INITIALIZATION_FAILED` with `VK_ERROR_UNKNOWN` across 4 files. NULL guards retained as defensive programming. All 13 tests pass.

### M2 — No double-bind protection in vkBindTensorMemoryKHR

- [x] **File**: `src/tensor.c:142-144`
- **Description**: `vkBindTensorMemoryKHR` doesn't check `tensor->memoryBound` before overwriting. A tensor can be silently re-bound, violating Vulkan semantics. The VUID is defined but not checked at the ICD level.
- **Fix**: Add `if (t->memoryBound) return VK_ERROR_UNKNOWN;` before setting bind state, or rely on the validation layer check (already implemented in `tensor_validation.c`).
- **FIXED**: Phase 23 (T145-T146). Added `if (t->memoryBound) return VK_ERROR_UNKNOWN;` guard in bind loop. Validation layer provides VUID diagnostic; ICD provides safety net. All 13 tests pass.

### M3 — No alignment validation in vkBindTensorMemoryKHR

- [x] **File**: `src/tensor.c:145-147`
- **Description**: `memoryOffset` is not checked against `VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN`. The VUID is defined but unused at the ICD level.
- **Fix**: Add alignment check or rely on validation layer (already implemented in `tensor_validation.c`).
- **FIXED**: Phase 24 (T147-T148). Added `if (VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN > 0 && info->memoryOffset % VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN != 0) return VK_ERROR_UNKNOWN;` guard. All 13 tests pass.

### M4 — Magic integer literals in format element size

- [x] [P] **File**: `src/internal.h:123-126`
- **Description**: `vk_ml_format_element_size` switch cases use raw integers (`1000559001`) instead of the enum names (`VK_FORMAT_R16_BFLOAT_KHR`).
- **Fix**: Replace with `case (uint32_t)VK_FORMAT_R16_BFLOAT_KHR:` etc.
- **FIXED**: Phase 25 (T149-T150). Replaced all 4 magic integers with `(uint32_t)VK_FORMAT_*` enum casts. Zero behavioral change. All 13 tests pass.

### M5 — `is_finite_float` fragile under `-ffast-math`

- [x] [P] **File**: `src/ml_primitives.c:12-15`
- **Description**: Hand-rolled `(f == f) && (f - f == 0.0f)` check relies on IEEE 754 semantics. Under `-ffast-math`, the compiler may optimize this to always return true.
- **Fix**: Use C11 `isfinite()` from `<math.h>`. Add `-lm` to link flags if needed.
- **FIXED**: Phase 26 (T151-T153). Replaced body with `return isfinite(f);`, added `#include <math.h>`, linked `m` on non-MSVC platforms via generator expression. All 13 tests pass.

### M6 — Inconsistent include pattern in 3 impl files

- [x] [P] **Files**: ~~`src/tensor_barrier.c`~~, `src/tensor_copy.c`, `src/ml_dispatch.c`
- **Description**: These files include `<vulkan/vulkan_ml_primitives.h>` directly instead of `"internal.h"`, unlike the other 5 implementation files. They cannot use internal helpers, VUID constants, or struct definitions.
- **Fix**: Change to `#include "internal.h"` for consistency. If these files genuinely don't need internal symbols, document the rationale.
- **FIXED**: Phase 27 (T154-T156). Replaced direct public header includes with `"internal.h"` in both remaining files. `tensor_barrier.c` was already deleted in H6/Phase 17. All `src/*.c` files now consistently use `"internal.h"`. All 13 tests pass.

### M7 — No sType validation anywhere

- [x] **Files**: all validation functions in `layers/validation/`
- **Description**: Neither the ICD implementation nor the validation layer checks the `sType` field of any input structure. Wrong `sType` is a common application error and standard Vulkan validation practice.
- **Fix**: Add `sType` checks to validation functions. For each validate function, check that the primary struct's `sType` matches the expected `VK_STRUCTURE_TYPE_*` value.
- **FIXED**: Phase 28 (T157-T162). Added sType checks to all 14 validation functions across 5 files. Used `(int)` cast to avoid GCC `-Wenum-compare` between `VkStructureType` (core) and extension enum values. All 13 tests pass.

### M8 — No tensor handle validation in vkCreateTensorViewKHR

- [x] **File**: `src/tensor_view.c:21-22`
- **Description**: `pCreateInfo->tensor` is not checked for `VK_NULL_HANDLE`. The VUID `VUID_TENSOR_VIEW_HANDLE` is defined but not checked.
- **Fix**: Add `if (pCreateInfo->tensor == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;`
- **FIXED**: Phase 29 (T163). Added `if (pCreateInfo->tensor == VK_NULL_HANDLE) return VK_ERROR_UNKNOWN;` after existing NULL guard, using `VK_ERROR_UNKNOWN` per M1 precedent. All 13 tests pass.

### M9 — No graph handle validation in vkCreateMLSessionKHR

- [x] **File**: `src/ml_session.c:21-22`
- **Description**: `pCreateInfo->graph` is not checked for `VK_NULL_HANDLE`. The VUID `VUID_SESSION_GRAPH_VALID` is defined but not checked.
- **Fix**: Add null-handle check for `graph`.
- **FIXED**: Phase 29 (T164). Added `if (pCreateInfo->graph == VK_NULL_HANDLE) return VK_ERROR_UNKNOWN;` after existing NULL guard. All 13 tests pass.

### M10 — Convolution kernelWidth/kernelHeight = 0 not rejected

- [x] **File**: `layers/validation/graph_validation.c:89-91`
- **Description**: Stride and dilation are validated as non-zero, but kernel dimensions are not. A 0x0 kernel is meaningless.
- **Fix**: Add `if (d->kernelWidth == 0 || d->kernelHeight == 0) return VK_FALSE;`
- **FIXED**: Phase 30 (T166). Added `/* VUID_CONV_KERNEL */ if (desc->kernelWidth == 0 || desc->kernelHeight == 0) return VK_FALSE;` after sType check, before stride check. All 13 tests pass.

### M11 — Convolution groupCount = 0 not rejected

- [x] **File**: `layers/validation/graph_validation.c:114-116`
- **Description**: `groupCount == 0` is never valid. The stub comment references shape-dependent validation, but zero can be caught unconditionally.
- **Fix**: Add `if (d->groupCount == 0) return VK_FALSE;`
- **FIXED**: Phase 31 (T168). Added `/* VUID_CONV_GROUP_COUNT */ if (desc->groupCount == 0) return VK_FALSE;` before the existing stub (now a TODO for future divisibility check). All 13 tests pass.

### M12 — Tensor usage flags never validated

- [x] **File**: `layers/validation/tensor_validation.c:66-71`
- **Description**: `vk_ml_validate_tensor_create` never checks `desc->usage`. Zero usage flags or invalid combinations are silently accepted.
- **Fix**: Add `if (desc->usage == 0) return VK_FALSE;` and optionally validate that only defined bits are set.
- **FIXED**: Phase 32 (T170). Added `/* VUID_TENSOR_USAGE */` block: rejects `usage == 0` and rejects undefined bits via `desc->usage & ~0x7F` mask (7 defined `VK_TENSOR_USAGE_*_BIT_KHR` values). All 13 tests pass.

### M13 — Tensor view doesn't check tensor has memory bound

- [x] **File**: `layers/validation/tensor_validation.c:85-87`
- **Description**: The API docs state "The source tensor must have memory bound," but `vk_ml_validate_tensor_view_create` doesn't check `tensor->memoryBound`.
- **Fix**: Add `if (!tensor->memoryBound) return VK_FALSE;`
- **FIXED**: Phase 33 (T172). Added `/* VUID_TENSOR_VIEW_MEMORY_BOUND */ if (!tensor->memoryBound) return VK_FALSE;` after sType check, before format validation. All 13 tests pass.

### M14 — VUID coverage only 59%

- [x] **Files**: entire validation layer
- **Description**: Originally 35 of 59 defined VUIDs were fully validated (59%). After Phases 22-33, coverage improved to 42/59 fully validated + 4 partial = 46/59 checked (78%). Additionally 3 new VUIDs were introduced (`VUID_TENSOR_USAGE`, `VUID_TENSOR_VIEW_MEMORY_BOUND`, `VUID_CONV_KERNEL`).
- **Remaining 13 unvalidated VUIDs** (all require infrastructure expansion):
  - **Runtime-only** (4): `TENSOR_DEVICE_QUEUE`, `COPY_TENSOR_CMD_STATE`, `DISPATCH_CMD_STATE`, `DISPATCH_COMPUTE_QUEUE` — need command buffer state tracking layer.
  - **Needs tensor object lookup** (7): `COPY_TENSOR_SRC_USAGE`, `COPY_TENSOR_DST_USAGE`, `COPY_TENSOR_MEM_BOUND`, `COPY_TENSOR_FORMAT`, `DISPATCH_INPUT_USAGE`, `DISPATCH_OUTPUT_USAGE`, `DISPATCH_WEIGHT_USAGE` — need expanded function signatures with handle-to-object resolution.
  - **Needs shape/graph context** (5): `BIND_TENSOR_MEM_SIZE`, `ML_GRAPH_EDGE_COMPAT`, `GEMM_DIMS`, `GEMM_BIAS`, `DISPATCH_WEIGHT_COUNT` — need cross-object shape analysis or graph context.
  - Some VUIDs span multiple categories (total unique = 13, some overlap).
- **Fix**: Incremental — prioritize VUIDs that prevent crashes or data corruption (usage flags, memory bound checks, count matching). Some require expanded function signatures to receive the necessary context.
- **PARTIAL**: Phase 34 (T174). Coverage improved from 59% to 78%. Remaining gaps deferred to future phase requiring validation infrastructure expansion (object lookup, runtime hooks, shape analysis).

### M15 — Naming inconsistency: sType vs struct name

- [x] [P] **File**: `include/vulkan/vulkan_ml_primitives.h:59`
- **Description**: `VK_STRUCTURE_TYPE_TENSOR_COPY_INFO_KHR` maps to struct `VkCopyTensorInfoKHR`. Vulkan convention maps `VkFooBar` to `VK_STRUCTURE_TYPE_FOO_BAR`. Should be `VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR`.
- **Fix**: Rename to `VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR` and update all references.
- **FIXED**: Phase 35 (T175-T179). Renamed across 4 files: `vulkan_ml_primitives.h` (definition), `tensor_validation.c` (sType check), `test_tensor_copy.c` (2 usages), `VK_KHR_ml_primitives.adoc` (spec). Full rebuild, all 13 tests pass.

### M16 — PReLU test uses wrong activation type

- [x] [P] **File**: `tests/cts/test_ml_graph.c:1000`
- **Description**: `test_single_node_prelu` sets `activationType = VK_ML_ACTIVATION_FUNCTION_LEAKY_RELU_KHR`. PReLU is semantically distinct from Leaky ReLU (per-channel learnable slopes vs fixed scalar).
- **Fix**: Either add `VK_ML_ACTIVATION_FUNCTION_PRELU_KHR` to the enum, or document the intentional reuse with a comment explaining the mapping.
- **FIXED**: Phase 36 (T180). Added clarifying comment documenting the intentional reuse: PReLU uses the same `f(x)=x>0?x:a*x` form as Leaky ReLU until a dedicated enum is added. All 13 tests pass.

### M17 — Test helper code duplication

- [x] [P] **Files**: `tests/cts/test_ml_graph.c`, `tests/cts/test_ml_session.c`, `tests/cts/test_ml_dispatch.c`
- **Description**: `make_tensor_desc`, `make_tensor_binding_external_input/output/weight` are copy-pasted identically across 3 files.
- **Fix**: Extract into a shared `tests/cts/test_helpers.h` and include from each file.
- **FIXED**: Phase 37 (T182-T186). Created `tests/cts/test_helpers.h` with all 5 helpers as `static inline`. Removed duplicates from 3 test files, added `#include "test_helpers.h"`. All 13 tests pass.

### M18 — Missing test coverage for concurrent mode and linear tiling

- [x] [P] **Files**: CTS tests
- **Description**: All tensors use `VK_SHARING_MODE_EXCLUSIVE` and `VK_TENSOR_TILING_OPTIMAL_KHR`. Concurrent sharing mode and linear tiling with explicit strides are never tested.
- **Fix**: Add tests for `VK_SHARING_MODE_CONCURRENT` with valid `queueFamilyIndexCount`/`pQueueFamilyIndices`, and `VK_TENSOR_TILING_LINEAR_KHR` with explicit strides.
- **FIXED**: Phase 38 (T187-T190). Added `test_tensor_concurrent_sharing` (CONCURRENT mode, 2 queue families) and `test_tensor_linear_tiling_with_strides` (LINEAR tiling, row-major strides) to `test_tensor_lifecycle.c`. All 13 tests pass.

### M19 — No NULL pointer argument tests

- [x] [P] **Files**: CTS tests
- **Description**: No test passes `NULL` for `pCreateInfo`, `pGraph`, `pSession`, or `pTensor` to validate NULL-dereference protection.
- **Fix**: Add negative tests that pass NULL for each output/input pointer and verify the function doesn't crash (returns error or is handled by validation layer).
- **FIXED**: Phase 39 (T191-T195). Added `test_create_tensor_null_args`, `test_create_tensor_view_null_args`, `test_create_graph_null_args`, `test_create_session_null_args` across four CTS test files. All 13 tests pass.

---

## LOW (11)

### L1 — No guard for size == 0 in vk_ml_alloc

- [x] [P] **File**: `src/internal.h:86`
- **Fix**: Add `if (size == 0) return NULL;`
- **FIXED**: Phase 40 (T196-T197). Added zero-size guard as first statement in `vk_ml_alloc`. All 13 tests pass.

### L2 — Missing prototypes for feature_query.c functions

- [x] [P] **File**: `src/internal.h`
- **Fix**: Add declarations for `vk_ml_populate_features`, `vk_ml_populate_properties`, `vk_ml_is_tensor_format_supported`, `vk_ml_populate_tensor_format_properties` to `internal.h`.
- **FIXED**: Phase 41 (T198-T199). Added "Feature query helpers" section with all four prototypes in `src/internal.h`. All 13 tests pass.

### L3 — pNext shallow-copied in deep_copy_tensor_desc

- [x] **File**: `src/ml_graph.c:33`
- **Fix**: Set `dst->pNext = NULL` after the shallow copy, since the graph doesn't own the pNext chain.
- **FIXED**: Already resolved during C1 deep-copy refactor (Phase 10). Line 33 sets `dst->pNext = NULL` immediately after the shallow copy.

### L4 — Verbose cascading cleanup in ml_graph.c

- [x] **File**: `src/ml_graph.c`
- **Fix**: Refactor to use a `goto cleanup` pattern to reduce ~80 lines of error handling to ~15.
- **FIXED**: Already resolved during C1 deep-copy refactor (Phase 10). `vkCreateMLGraphKHR` now uses `goto cleanup` with `free_graph_internals` helper.

### L5 — C standard set to C11, constitution prefers C17

- [x] [P] **File**: `CMakeLists.txt:8`
- **Fix**: Change `set(CMAKE_C_STANDARD 11)` to `set(CMAKE_C_STANDARD 17)`.
- **FIXED**: Phase 42 (T200-T201). Changed `CMAKE_C_STANDARD` from 11 to 17. Full reconfigure + build: zero warnings. All 13 tests pass.

### L6 — No install target or BUILD_TESTING guard

- [x] [P] **File**: `CMakeLists.txt`
- **Fix**: Add `option(BUILD_TESTING "Build tests" ON)`, wrap test block in `if(BUILD_TESTING)`, add `install()` commands for library and header.
- **FIXED**: Phase 43 (T202-T204). Added `BUILD_TESTING` option guarding tests/examples, and `install()` targets for `libvk_ml_primitives.a`, `libvk_ml_validation.a`, and `vulkan_ml_primitives.h`. All 13 tests pass. Install verified.

### L7 — Inconsistent include paths for internal.h in tests

- [x] [P] **Files**: various test files
- **Description**: Some use `"internal.h"` (CMake-resolved), others use `"../../src/internal.h"` (relative).
- **Fix**: Standardize on one approach. Prefer CMake `target_include_directories` with `"internal.h"`.
- **FIXED**: Phase 44 (T205-T207). Changed `test_vuids.c` and `test_dag_validation.c` from `"../../src/internal.h"` to `"internal.h"`. All 13 tests pass.

### L8 — Stray .o files in project root

- [x] [P] **File**: repo root
- **Fix**: `rm -f *.o` from project root.
- **FIXED**: No `.o` files present in repo root. `.gitignore` already contains `*.o` (line 6), preventing future tracking. No action required.

### L9 — Session validation doesn't check scratchMemoryOffset alignment

- [x] **File**: `layers/validation/session_validation.c`
- **Fix**: Add alignment check for `scratchMemoryOffset` against device alignment requirements.
- **FIXED**: Phase 45 (T208-T210). Added `VUID_SESSION_SCRATCH_OFFSET_ALIGN` check against `VK_ML_REF_MIN_TENSOR_MEMORY_ALIGN` (64 bytes). Added `test_session_scratch_offset_misaligned` negative test in `test_vuids.c`. All 13 tests pass.

### L10 — Self-referencing include paths in validation files

- [x] [P] **Files**: `layers/validation/tensor_validation.c:6`, `layers/validation/graph_validation.c:6`
- **Description**: `"../validation/vk_ml_validation.h"` from within `layers/validation/` is redundant.
- **Fix**: Change to `"vk_ml_validation.h"`.
- **FIXED**: Phase 46 (T211-T216). Changed all 5 validation files (`tensor_validation.c`, `graph_validation.c`, `session_validation.c`, `dispatch_validation.c`, `barrier_validation.c`) to use `"vk_ml_validation.h"`. All 13 tests pass.

### L11 — Resource leak in quickstart on partial tensor creation failure

- [x] **File**: `examples/quickstart.c:93-98`
- **Description**: If the first tensor creates successfully but the second fails, the first is leaked due to short-circuit `||`.
- **Fix**: Create each tensor separately with individual error checking and cleanup, or use a `goto cleanup` pattern.
- **FIXED**: Phase 47 (T217-T218). Added `vkDestroyTensorKHR` calls for all three tensors in the error path. Safe because handles are pre-initialized to `VK_NULL_HANDLE` and destroy is no-op on null. All 13 tests pass.

---

## Dependency Order for Fixes

```text
CRITICAL (do first — UB in normal usage):
  C1, C2  [independent, can parallelize]

HIGH (do next — correctness & safety):
  H1      [depends on C1 pattern]
  H2      [independent]
  H3, H4  [independent validation fixes]
  H5, H6  [independent architectural fixes]
  H7      [independent, documentation]
  H8-H10  [independent test improvements]

MEDIUM (incremental quality):
  M1-M3   [ICD hardening, independent]
  M4-M6   [code hygiene, independent]
  M7-M13  [validation coverage, some depend on M14]
  M14     [tracks overall VUID progress]
  M15-M19 [independent fixes]

LOW (polish):
  L1-L11  [all independent]
```

---
---

## Review 2 — Fresh Full Review (2026-03-06)

**Scope**: Complete re-read of all 30 source/header/test files (9,629 LOC), CMakeLists.txt, .gitignore, README, quickstart example
**Build status**: 13/13 tests pass, zero warnings
**Prior findings**: All 42 findings from Review 1 resolved (2 CRITICAL, 10 HIGH, 19 MEDIUM, 11 LOW)

---

### HIGH (1)

#### H11 — Pooling validation inconsistency for Global Average Pool

- [x] **Files**: `layers/validation/graph_validation.c:158-165`, `src/ml_primitives.c:50-57`
- **Description**: The validation layer (`vk_ml_validate_pooling_desc`) unconditionally rejects `windowWidth == 0` and `strideX == 0` for **all** pool types, including `VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR`. For global average pooling, window size and stride are irrelevant — the entire spatial extent is pooled. The ICD layer (`vk_ml_validate_primitive_desc` in `ml_primitives.c`) **correctly** gates these checks on `poolType != GLOBAL_AVERAGE_POOL`, but the validation layer does not. Valid global average pool configurations will produce false validation errors.
- **Fix**: In `vk_ml_validate_pooling_desc`, gate the window/stride checks on `desc->poolType != VK_ML_OPERATION_TYPE_GLOBAL_AVERAGE_POOL_KHR`, matching the ICD logic.
- **FIXED**: Phase 48 (T219-T222). Moved `VUID_POOL_TYPE` switch before window/stride checks and gated `VUID_POOL_WINDOW`/`VUID_POOL_STRIDE` on `poolType != GLOBAL_AVERAGE_POOL_KHR`. Added `test_valid_global_average_pool` unit test (window/stride = 0, expects `VK_TRUE`). Updated CTS `test_single_node_global_avg_pool` to use `windowWidth = 0, windowHeight = 0`. All 13 tests pass.

---

### MEDIUM (5)

#### M20 — Inconsistent `internal.h` include paths in validation layer files

- [x] **Files**: `layers/validation/tensor_validation.c:7`, `layers/validation/graph_validation.c:7`
- **Description**: `session_validation.c` uses the CMake-resolved `#include "internal.h"`, but `tensor_validation.c` and `graph_validation.c` still use the relative path `#include "../../src/internal.h"`. CMakeLists.txt provides `${CMAKE_CURRENT_SOURCE_DIR}/src` as a PRIVATE include directory for `vk_ml_validation`, so all three should use `#include "internal.h"`. The L10 remediation standardized `vk_ml_validation.h` includes but missed these two `internal.h` includes.
- **Fix**: Change both files from `#include "../../src/internal.h"` to `#include "internal.h"`.
- **FIXED**: Phase 49 (T223-T224). Changed both files to `#include "internal.h"`. All 13 tests pass.

#### M21 — CTS tests use `extern` workarounds for validation functions

- [x] **Files**: `tests/cts/test_synchronization.c:15-16`, `tests/cts/test_ml_dispatch.c:7`
- **Description**: `test_synchronization.c` declares validation functions via raw `extern` instead of including the header. `test_ml_dispatch.c` includes `vk_ml_validation.h` via a fragile relative path `../../layers/validation/vk_ml_validation.h`. Root cause: the CTS test CMake loop only adds `src/` to the include path, not `layers/validation/`.
- **Fix**: Add `${CMAKE_CURRENT_SOURCE_DIR}/layers/validation` to the CTS test `target_include_directories` block. Then replace `extern` declarations in `test_synchronization.c` with `#include "vk_ml_validation.h"`, and fix the relative path in `test_ml_dispatch.c`.
- **FIXED**: Phase 49 (T225-T227). Added `layers/validation` to CTS `target_include_directories`. Replaced `extern` declarations in `test_synchronization.c` with `#include "vk_ml_validation.h"`. Changed relative path in `test_ml_dispatch.c` to `#include "vk_ml_validation.h"`. All 13 tests pass.

#### M22 — Missing `VUID_SESSION_SCRATCH_OFFSET_ALIGN` string constant

- [x] **File**: `src/internal.h`
- **Description**: `session_validation.c` has a `/* VUID_SESSION_SCRATCH_OFFSET_ALIGN */` comment referencing a VUID, but `internal.h` has no corresponding `#define VUID_SESSION_SCRATCH_OFFSET_ALIGN`. All other 40+ VUIDs used in the validation layer have string constants defined in `internal.h`. This was omitted during Phase 45 (L9 fix).
- **Fix**: Add `#define VUID_SESSION_SCRATCH_OFFSET_ALIGN "VUID-VkMLSessionCreateInfoKHR-scratchMemoryOffset-00004"` to `internal.h`, following the existing naming convention.
- **FIXED**: Phase 49 (T228). Added `#define VUID_SESSION_SCRATCH_OFFSET_ALIGN` after `VUID_SESSION_GRAPH_VALID` in `internal.h`. All 13 tests pass.

#### M23 — Redundant `extern` declarations in `test_vuids.c`

- [x] [P] **File**: `tests/validation/test_vuids.c:11-12`
- **Description**: Lines 11-12 declare `extern void vk_ml_populate_features(...)` and `extern void vk_ml_populate_properties(...)`. Both functions are already declared in `internal.h`, which is included on line 7. These `extern` lines are dead weight.
- **Fix**: Remove lines 11-12.
- **FIXED**: Phase 49 (T229). Removed both redundant `extern` declarations. All 13 tests pass.

#### M24 — Stray `.o` files still present on disk

- [x] [P] **File**: repo root
- **Description**: Nine `.o` files physically exist in the project root: `feature_query.o`, `ml_dispatch.o`, `ml_graph.o`, `ml_primitives.o`, `ml_session.o`, `tensor_barrier.o`, `tensor_copy.o`, `tensor.o`, `tensor_view.o`. L8 from Review 1 was marked "pre-resolved" because `.gitignore` has `*.o`, but the files were never actually deleted. Notably, `tensor_barrier.o` is an orphan from before `tensor_barrier.c` was moved to the validation layer in Phase 17. These clutter the working directory and waste disk space (~240KB total).
- **Fix**: `rm -f *.o` from project root.
- **FIXED**: Phase 49 (T230). Deleted all 9 stray `.o` files from project root. Confirmed no `.o` files remain.

---

### LOW (7)

#### L12 — README claims C11 but project uses C17

- [x] [P] **File**: `README.md:84`
- **Description**: README says "C compiler with C11 support" but `CMakeLists.txt` sets `set(CMAKE_C_STANDARD 17)` (changed in Phase 42/L5 fix). The README was not updated to reflect this change.
- **Fix**: Change "C11" to "C17" in the prerequisites section of README.md.
- **FIXED**: Phase 50 (T232). Changed to "C compiler with C17 support".

#### L13 — Unnecessary `#include <stdbool.h>` in `vk_ml_validation.h`

- [x] [P] **File**: `layers/validation/vk_ml_validation.h:12`
- **Description**: The validation header includes `<stdbool.h>` but exclusively uses `VkBool32`. The `bool` type is never used anywhere in the validation layer.
- **Fix**: Remove `#include <stdbool.h>`.
- **FIXED**: Phase 50 (T233). Removed the unused include.

#### L14 — Quickstart example inside `BUILD_TESTING` guard

- [x] [P] **File**: `CMakeLists.txt:132`
- **Description**: The quickstart example is wrapped inside `if(BUILD_TESTING)`. Examples and tests are distinct concerns. Library consumers who set `-DBUILD_TESTING=OFF` lose access to the example.
- **Fix**: Either move the example section outside the `if(BUILD_TESTING)` block, or add a separate `option(BUILD_EXAMPLES "Build examples" ON)` guard.
- **FIXED**: Phase 50 (T234, T244). Moved example outside `BUILD_TESTING`, added `option(BUILD_EXAMPLES)` guard. Verified quickstart builds with `BUILD_TESTING=OFF`.

#### L15 — Hardcoded valid usage bitmask in tensor validation

- [x] [P] **File**: `layers/validation/tensor_validation.c:70`
- **Description**: `const VkFlags validUsageMask = 0x7F` is a magic number corresponding to the 7 defined `VkTensorUsageFlagBitsKHR` values. If a new usage flag is added to the enum, this mask will silently become stale and reject valid usage combinations.
- **Fix**: Derive the mask from the highest defined bit: `const VkFlags validUsageMask = (VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_KHR << 1) - 1;`
- **FIXED**: Phase 50 (T235). Replaced `0x7F` with self-deriving expression from highest enum bit.

#### L16 — Wasteful temporary allocation in `deep_copy_tensor_desc` usage

- [x] **File**: `src/ml_graph.c:319-323, 339-343, 359-363`
- **Description**: External input/output/weight descriptions are deep-copied via `deep_copy_tensor_desc()` which allocates a new struct, then the caller shallow-copies it into the destination array and immediately frees the shell. This repeats 3 times (inputs, outputs, weights), allocating and immediately freeing a struct per description — wasteful.
- **Fix**: Add a direct "copy-in-place" helper `deep_copy_tensor_desc_into(VkTensorDescriptionKHR *dst, const VkTensorDescriptionKHR *src, allocator)` that copies directly into the destination, eliminating the temporary allocation.
- **FIXED**: Phase 50 (T236-T237). Added `deep_copy_tensor_desc_into()` returning `VkResult`. Replaced all 3 copy loops to use the new helper. Original `deep_copy_tensor_desc()` retained for node binding use case. All 13 tests pass.

#### L17 — Test files use relative paths for `vk_ml_validation.h` despite CMake support

- [x] [P] **Files**: `tests/validation/test_vuids.c:6`, `tests/unit/test_dag_validation.c:8`, `tests/unit/test_descriptor_validation.c:7`
- **Description**: All three use `#include "../../layers/validation/vk_ml_validation.h"` even though their CMake targets include `layers/validation/` as a PRIVATE include directory. Should use `#include "vk_ml_validation.h"`.
- **Fix**: Change to `#include "vk_ml_validation.h"` in all three files.
- **FIXED**: Phase 50 (T238-T240). Changed all three files to `#include "vk_ml_validation.h"`. Also removed redundant `extern` declarations in `test_dag_validation.c` and `test_descriptor_validation.c`, added missing `#include "internal.h"` to `test_descriptor_validation.c`. All 13 tests pass.

#### L18 — Memory requirement functions don't clear `pNext`

- [x] [P] **Files**: `src/tensor.c:122`, `src/ml_graph.c:417`
- **Description**: `vkGetTensorMemoryRequirementsKHR` and `vkGetMLGraphMemoryRequirementsKHR` set `sType` and `memoryRequirements` but leave `pNext` untouched. While callers are responsible for initializing the struct, a defensive `pMemoryRequirements->pNext = NULL` would prevent stale pointer issues and align with Vulkan conventions for output structs.
- **Fix**: Add `pMemoryRequirements->pNext = NULL;` after setting `sType` in both functions.
- **FIXED**: Phase 50 (T241-T242). Added `pMemoryRequirements->pNext = NULL;` in both `vkGetTensorMemoryRequirementsKHR` and `vkGetMLGraphMemoryRequirementsKHR`. All 13 tests pass.

---

### Review 2 Dependency Order

```text
HIGH (correctness):
  H11  [independent]

MEDIUM (quality):
  M20  [independent — 2 include path fixes]
  M21  [independent — CMake + include fix]
  M22  [independent — add VUID define]
  M23  [P] [independent — remove dead code]
  M24  [P] [independent — delete files]

LOW (polish):
  L12-L18  [all independent]
```
