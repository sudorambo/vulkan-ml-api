# Implementation Plan: L12–L18 Low-Severity Polish Batch

**Branch**: `001-ml-primitives` | **Date**: 2026-03-06 | **Spec**: Review 2 findings L12–L18
**Input**: `specs/001-ml-primitives/review-findings.md:444-484`

## Summary

Batch remediation of 7 independent LOW-severity findings from Review 2. These are polish-level changes: documentation accuracy, dead includes, build structure, magic number elimination, micro-optimization of allocation patterns, include path consistency, and defensive output struct initialization. No behavioral changes to the API.

## Technical Context

**Language/Version**: C17 (`CMAKE_C_STANDARD 17`)
**Primary Dependencies**: Vulkan 1.3, `VK_KHR_ml_primitives` (internal ICD + validation layer)
**Storage**: N/A
**Testing**: CTest (13 existing tests — 10 CTS, 1 validation, 2 unit)
**Target Platform**: Linux x86_64 (current), cross-platform by design
**Project Type**: Library (ICD reference implementation + validation layer)
**Constraints**: Zero warnings, all 13 tests must pass after changes

## Constitution Check

*GATE: Must pass before proceeding.*

| Principle | Relevant? | Status | Notes |
|-----------|-----------|--------|-------|
| I. Spec-Driven | Low | PASS | No spec behavior changes. L15 improves adherence to spec enum values. |
| II. Vulkan C API | Yes | PASS | All changes maintain Vulkan naming/patterns. L18 aligns with Vulkan output struct conventions. |
| III. Portability | Low | PASS | No platform-specific changes. |
| IV. Test-First | Yes | PASS | No new behavior to test. Existing 13 tests verify no regressions. L16 requires care — new helper must not break existing deep-copy semantics. |
| V. Resource Lifecycle | Yes | PASS | L16 eliminates wasteful alloc/free cycle. L18 defensively clears output pointers. |
| VI. Backward Compat | N/A | PASS | Internal changes only. No API surface modification. |
| VII. Simplicity | Yes | PASS | All changes reduce complexity (remove dead code, eliminate magic numbers, simplify allocation patterns). |

No violations. No complexity tracking needed.

## Findings Analysis

### L12 — README claims C11 but project uses C17

- **File**: `README.md:84`
- **Current**: `"C compiler with C11 support"`
- **Target**: `"C compiler with C17 support"`
- **Risk**: None. Documentation-only change.

### L13 — Unnecessary `#include <stdbool.h>` in `vk_ml_validation.h`

- **File**: `layers/validation/vk_ml_validation.h:12`
- **Current**: `#include <stdbool.h>` present but `bool` never used (only `VkBool32`)
- **Target**: Remove the include
- **Risk**: None. Verified no `bool` usage in any validation layer source file.

### L14 — Quickstart example inside `BUILD_TESTING` guard

- **File**: `CMakeLists.txt:130-137`
- **Current**: `add_executable(quickstart ...)` is inside `if(BUILD_TESTING)` block
- **Target**: Move example build outside `if(BUILD_TESTING)` block, add separate `option(BUILD_EXAMPLES "Build example programs" ON)` guard
- **Risk**: Low. The quickstart target is independent — it only links `vk_ml_primitives`, not `vk_ml_validation`.

### L15 — Hardcoded valid usage bitmask in tensor validation

- **File**: `layers/validation/tensor_validation.c:70`
- **Current**: `const VkFlags validUsageMask = 0x7F;`
- **Target**: `const VkFlags validUsageMask = (VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_KHR << 1) - 1;`
- **Risk**: None. `VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_KHR = 0x40`, so `(0x40 << 1) - 1 = 0x7F`. Identical result today, but self-updating if new flags are added above this bit.

### L16 — Wasteful temporary allocation in `deep_copy_tensor_desc` usage

- **File**: `src/ml_graph.c:319-363`
- **Current**: Three loops (inputs, outputs, weights) each call `deep_copy_tensor_desc()` which allocates a new `VkTensorDescriptionKHR` shell, then the caller does `graph->descs[i] = *copy; vk_ml_free(pAllocator, copy);` — allocating and immediately freeing the shell struct 3× per description.
- **Target**: Add `deep_copy_tensor_desc_into(VkTensorDescriptionKHR *dst, const VkTensorDescriptionKHR *src, const VkAllocationCallbacks *pAllocator)` that copies directly into `dst`, skipping the shell allocation. The existing `deep_copy_tensor_desc()` remains for callers that need an allocated copy (node bindings at line 143).
- **Risk**: Medium-low. Must preserve identical deep-copy semantics (pNext=NULL, pDimensions copy, pStrides copy). Must handle OOM identically. The existing function at line 22 remains unchanged for backward compat with the node binding use case.

### L17 — Test files use relative paths for `vk_ml_validation.h`

- **Files**: `tests/validation/test_vuids.c:6`, `tests/unit/test_dag_validation.c:8`, `tests/unit/test_descriptor_validation.c:7`
- **Current**: All use `#include "../../layers/validation/vk_ml_validation.h"`
- **Target**: `#include "vk_ml_validation.h"` (CMake already provides the include directory for all three targets)
- **Bonus**: `test_dag_validation.c:11-12` and `test_descriptor_validation.c:12` have redundant `extern` declarations for `vk_ml_populate_features`/`vk_ml_populate_properties` that are already in `internal.h` (included). Remove them.
- **Risk**: None. CMake `target_include_directories` already configured correctly for all three targets.

### L18 — Memory requirement functions don't clear `pNext`

- **Files**: `src/tensor.c:122`, `src/ml_graph.c:417`
- **Current**: Both set `sType` and `memoryRequirements` but leave `pNext` untouched
- **Target**: Add `pMemoryRequirements->pNext = NULL;` after setting `sType` in both functions
- **Risk**: None. Defensive initialization. Callers that chain `pNext` do so before the call and expect it to be preserved — but Vulkan convention for output-only structs is to write all fields. This is an output struct, not input/output.

## Affected Files

```text
README.md                                          L12 — doc fix
layers/validation/vk_ml_validation.h               L13 — remove stdbool.h
CMakeLists.txt                                     L14 — move example out of BUILD_TESTING
layers/validation/tensor_validation.c              L15 — derive usage mask from enum
src/ml_graph.c                                     L16 — add copy-in-place helper, L18 — clear pNext
src/tensor.c                                       L18 — clear pNext
tests/validation/test_vuids.c                      L17 — fix include path
tests/unit/test_dag_validation.c                   L17 — fix include path + remove externs
tests/unit/test_descriptor_validation.c            L17 — fix include path + remove extern
```

## Test Plan

No new tests required — these are non-behavioral changes. Validation:

1. **Build**: `cmake --build build` — zero warnings
2. **Tests**: `ctest --output-on-failure` — all 13 tests pass
3. **Manual verification**: Confirm `cmake -B build -S . -DBUILD_TESTING=OFF` still builds the quickstart example (L14 validation)

## Project Structure

No new files or directories. All changes are in existing files.

```text
specs/001-ml-primitives/
├── plan.md              # This file
└── tasks.md             # Phase 50 tasks (generated by /speckit.tasks)
```
