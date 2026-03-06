# Implementation Plan: M20–M24 — Code Hygiene Batch (Review 2 MEDIUMs)

**Branch**: `001-ml-primitives` | **Date**: 2026-03-06 | **Spec**: `spec/VK_KHR_ml_primitives.adoc`
**Input**: Review findings M20–M24 from `specs/001-ml-primitives/review-findings.md:405-433`

## Summary

Batch remediation of 5 MEDIUM-severity code hygiene findings from Review 2. All are independent, low-risk changes that improve include consistency, remove dead code, and clean up stray build artifacts. No behavioral changes.

| Finding | Summary | Files |
|---------|---------|-------|
| M20 | Validation layer `internal.h` include paths inconsistent | `tensor_validation.c`, `graph_validation.c` |
| M21 | CTS tests use `extern` / relative paths for validation funcs | `CMakeLists.txt`, `test_synchronization.c`, `test_ml_dispatch.c` |
| M22 | Missing `VUID_SESSION_SCRATCH_OFFSET_ALIGN` define | `src/internal.h` |
| M23 | Redundant `extern` declarations in `test_vuids.c` | `tests/validation/test_vuids.c` |
| M24 | Stray `.o` files on disk | repo root |

## Technical Context

**Language/Version**: C17
**Primary Dependencies**: Vulkan 1.3, `VK_KHR_ml_primitives` extension
**Storage**: N/A
**Testing**: CTest, custom test harness
**Target Platform**: Linux x86_64 (dev), cross-platform Vulkan 1.3
**Project Type**: Library (reference ICD + validation layer)
**Constraints**: Zero warnings under `-Wall -Wextra -Wpedantic -Werror`
**Scale/Scope**: 7 files changed, ~10 lines modified + 9 `.o` files deleted

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Spec-Driven Development | PASS | No spec changes; VUID define is traceability improvement |
| II. Vulkan C API Conventions | PASS | No API surface changes |
| III. Portability | PASS | Replacing fragile relative paths with CMake-resolved includes improves portability |
| IV. Test-First with Validation Layers | PASS | No test behavior changes; tests remain identical |
| V. Explicit Resource Lifecycle | N/A | No lifecycle changes |
| VI. Backward Compatibility | PASS | No behavioral changes |
| VII. Simplicity | PASS | Removes unnecessary code (extern decls, stray files) |

**All gates pass. No violations.**

## Project Structure

### Affected files

```text
layers/validation/tensor_validation.c   # M20: fix internal.h include path
layers/validation/graph_validation.c    # M20: fix internal.h include path
CMakeLists.txt                          # M21: add layers/validation to CTS include dirs
tests/cts/test_synchronization.c        # M21: replace extern with #include
tests/cts/test_ml_dispatch.c            # M21: fix relative path include
src/internal.h                          # M22: add VUID define
tests/validation/test_vuids.c           # M23: remove redundant externs
*.o (repo root)                         # M24: delete stray object files
```

## Analysis

### M20 — Inconsistent `internal.h` include paths

`tensor_validation.c` and `graph_validation.c` use `#include "../../src/internal.h"` while `session_validation.c` already uses `#include "internal.h"`. CMakeLists.txt line 60 adds `${CMAKE_CURRENT_SOURCE_DIR}/src` as a PRIVATE include dir for `vk_ml_validation`, so the CMake-resolved path works. Simple find-and-replace in 2 files.

### M21 — CTS extern workarounds

Root cause: the CTS test loop (CMakeLists.txt:98-100) only adds `src/` to include dirs. `test_synchronization.c` works around this with raw `extern` declarations. `test_ml_dispatch.c` uses a fragile `../../layers/validation/vk_ml_validation.h` path. Fix: add `${CMAKE_CURRENT_SOURCE_DIR}/layers/validation` to the CTS test include dirs, then clean up both test files.

### M22 — Missing VUID define

All other VUIDs in `internal.h` have string constant defines. `VUID_SESSION_SCRATCH_OFFSET_ALIGN` was added as a comment in `session_validation.c` during Phase 45 but the corresponding `#define` was omitted. Insert it after `VUID_SESSION_GRAPH_VALID` (line 245) to maintain alphabetical/logical ordering.

### M23 — Redundant extern declarations

`test_vuids.c` lines 11-12 declare `extern` for `vk_ml_populate_features` and `vk_ml_populate_properties`, but both are already declared in `internal.h` (included on line 7). Remove the dead lines.

### M24 — Stray .o files

Nine `.o` files in the repo root from an old manual build. `.gitignore` has `*.o` so they're not tracked, but they physically exist and waste ~240KB. One (`tensor_barrier.o`) is orphaned from before Phase 17's file move. Simple `rm -f *.o`.
