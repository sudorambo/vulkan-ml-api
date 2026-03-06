# Quickstart: Implementing the v1.0 Readiness Fixes

**Plan**: [plan.md](plan.md) | **Detailed fixes**: [v1.0-readiness.plan.md](v1.0-readiness.plan.md)

This guide walks through implementing the 35 fixes needed for the 1.0 release,
organized by phase with verification steps between each phase.

## Prerequisites

- Repository cloned and on the `001-ml-primitives` branch
- Vulkan SDK 1.3+ installed
- CMake 3.20+, GCC 11+ or Clang 14+

```bash
git checkout 001-ml-primitives
cmake --preset default
cmake --build build
ctest --test-dir build --output-on-failure   # baseline: all current tests pass
```

## Phase 1: Public API Surface (3 fixes)

Start here — all other phases depend on correct header values.

### Step 1: Fix sType collisions

Edit `include/vulkan/vulkan_ml_primitives.h`. Change lines 73–76:

```c
VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR  = 1000559024,
VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR       = 1000559025,
VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR  = 1000559026,
VK_STRUCTURE_TYPE_TENSOR_COPY_KHR             = 1000559027,
```

Then search the entire codebase for references to the old values or these
enum names and verify the comparisons still work (they use the name, not
the number, so they auto-update).

### Step 2: Rename VK_FORMAT_R8_BOOL

In `vulkan_ml_primitives.h`, rename `VK_FORMAT_R8_BOOL` to `VK_FORMAT_R8_BOOL_KHR`.
Then update references in:
- `src/internal.h` (`vk_ml_format_element_size` switch case)
- `src/feature_query.c` (supported formats array)
- Any test files that reference this format

### Step 3: Move VkMLResizeModeKHR

Cut the `VkMLResizeModeKHR` enum definition and paste it into the Enumerations
section, after `VkMLTensorBindingTypeKHR`.

### Verify Phase 1

```bash
cmake --build build 2>&1 | head -50   # must compile cleanly
ctest --test-dir build --output-on-failure   # all tests pass
```

## Phase 2: ICD Fixes (8 fixes)

### Key changes

1. **ml_graph.c**: Add 4 cases to `op_desc_size_by_stype()` for concat/reshape/transpose/resize. Add deep-copy logic for reshape `pOutputDimensions` and transpose `pPermutation`. Guard scratch calculation loops with NULL checks.

2. **ml_session.c**: Add `memset(session, 0, sizeof(...))` after allocation. Add scratch size validation: `if (pCreateInfo->scratchMemorySize < g->scratchMemorySize) return VK_ERROR_UNKNOWN;`

3. **tensor.c**: Add sType validation. Change NULL handle `continue` to `return VK_ERROR_UNKNOWN`.

4. **tensor_view.c**: Add sType validation.

5. **tensor_copy.c**: Add `pExtents` NULL check. Replace `(int)` casts with `(uint32_t)`.

6. **ml_dispatch.c**: Replace `(int)` casts with `(uint32_t)`.

### Verify Phase 2

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

## Phase 3: Validation Layer (7 fixes)

### Key changes

1. **tensor_validation.c**: Fix overflow in view range check. Add `pExtents` NULL check. Add src/dst NULL handle checks in copy validation.

2. **graph_validation.c**: Add stack overflow guard. Implement `vk_ml_validate_activation_desc()`. Change unknown sType default to `return VK_FALSE`. Add cases for concat/reshape/transpose/resize.

3. **vk_ml_validation.h**: Declare `vk_ml_validate_activation_desc()`.

### Verify Phase 3

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

## Phase 4: Infrastructure (8 fixes)

1. **CMakeLists.txt**: Fix clang-tidy config, add NAMESPACE to export
2. **cmake/vk_ml_primitives.pc.in**: Use `@CMAKE_INSTALL_FULL_LIBDIR@`
3. **.gitignore**: Add `docs/html/` and `build-*/`
4. **.github/workflows/ci.yml**: Dynamic codename, format check, remove CXX compiler
5. **CONTRIBUTING.md**: Align compiler versions to GCC 11+, Clang 14+, MSVC 2022+

### Verify Phase 4

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

## Phase 5: Tests (6 new test functions)

Add to existing test files — no new executables needed:

1. `tests/unit/test_validation_coverage.c`: `test_tensor_view_uint32_overflow`, `test_tensor_copy_null_extents`, `test_tensor_copy_null_handles`
2. `tests/unit/test_descriptor_validation.c`: activation descriptor validation tests
3. `tests/unit/test_dag_validation.c`: `test_valid_diamond_dag`
4. `tests/cts/test_ml_graph.c`: concat/reshape/transpose/resize graph creation tests

### Verify Phase 5

```bash
cmake --build build
ctest --test-dir build --output-on-failure   # new tests visible and passing
```

## Phase 6: Release Polish (3 items)

1. Update `CHANGELOG.md`: move `[Unreleased]` to `[1.0.0]`, add v1.0 fixes
2. Bump version: `CMakeLists.txt` `VERSION 1.0.0`, `Doxyfile` `PROJECT_NUMBER = 1.0.0`
3. Update `README.md`: verify test counts, version references

### Final Verification

```bash
cmake --build build
ctest --test-dir build --output-on-failure
# Static analysis (if clang-tidy/cppcheck available):
cmake --build build --target cppcheck
# Format check:
find src include layers examples -name '*.c' -o -name '*.h' | xargs clang-format --dry-run --Werror
```

## Done

All 35 fixes applied. Tag the release:

```bash
git tag -a v1.0.0 -m "VK_KHR_ml_primitives v1.0.0"
```
