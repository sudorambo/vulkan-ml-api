# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] - 2026-03-06

### Added

- `VK_ML_REF_ENABLE_VALIDATION` CMake option to wire validation into ICD stubs at compile time

### Changed

- `vkGetTensorMemoryRequirementsKHR` now accounts for explicit `pStrides` (strided layout sizing)
- `vkBindTensorMemoryKHR` returns `VK_ERROR_VALIDATION_FAILED_EXT` for already-bound and misaligned-offset errors instead of generic `VK_ERROR_UNKNOWN`
- `VkMLSessionKHR_T` uses a dedicated `void *autoScratchHost` field instead of type-punning `VkDeviceMemory`
- cppcheck CI invocation includes `layers/validation` include path

## [1.0.0] - 2026-03-05

### Added

- GitHub Actions CI with GCC/Clang/MSVC matrix and static analysis
- CMake package config (`find_package(vk_ml_primitives)` support)
- pkg-config support (`vk_ml_primitives.pc`)
- CMake presets for default, release, and CI builds
- `.clang-format` and `.clang-tidy` configuration files
- `CONTRIBUTING.md` with development guidelines
- Comprehensive validation coverage tests (tensor view, copy, session, dispatch, graph per-node, boundaries, barriers)
- 22 missing VUID defines to complete spec coverage (81/81)
- Per-node primitive descriptor validation in graph creation
- Session scratch memory auto-allocation
- Tensor create sharing mode validation
- Tensor copy region offset validation
- Session null graph handle validation
- Enriched `vkCmdCopyTensorKHR` and `vkCmdDispatchMLGraphKHR` stubs with parameter validation
- `vk_ml_validate_primitive_desc` integration into `vkCreateMLGraphKHR`
- Doxygen configuration for API reference generation
- Concat, Reshape, Transpose, Resize primitive descriptor types in ICD and validation
- Activation descriptor validation (enum range, clamp param ordering)
- Deep-copy of pointer-containing descriptors (Reshape `pOutputDimensions`, Transpose `pPermutation`)
- sType validation for `vkCreateTensorKHR` and `vkCreateTensorViewKHR`
- Scratch memory size and offset alignment validation in session creation
- `clang-format` compliance check in CI
- Tests for integer overflow, NULL handles, activation validation, diamond DAG, new ops, deep-copy

### Changed

- ICD `vkCreateMLGraphKHR` now validates per-node primitive descriptors
- ICD `vkCreateMLSessionKHR` implements scratch auto-allocation when `scratchMemory == VK_NULL_HANDLE`
- Graph validation layer now delegates to per-descriptor validators
- Conv group count TODO resolved with deferred-to-dispatch note
- Reassigned 4 colliding `VkStructureType` values to unique range (1000559024–1000559027)
- Renamed `VK_FORMAT_R8_BOOL` to `VK_FORMAT_R8_BOOL_KHR` for KHR naming consistency
- Moved `VkMLResizeModeKHR` enum to the Enumerations section of the header
- `vkBindTensorMemoryKHR` now returns `VK_ERROR_UNKNOWN` for NULL handles instead of silently skipping
- Validation layer rejects unknown operation sTypes instead of silently accepting
- Replaced all `(int)` casts on sType comparisons with `(uint32_t)` across ICD and validation
- CMake exported targets now use `VulkanML::` namespace
- pkg-config `libdir` uses `@CMAKE_INSTALL_LIBDIR@` for relocatable installs
- clang-tidy defers to `.clang-tidy` config instead of inline overrides
- CI uses dynamic Ubuntu codename via `lsb_release -cs`
- Compiler requirements updated to GCC 12+, Clang 15+, MSVC 2022+

### Fixed

- Critical: 4 duplicate `VkStructureType` values causing undefined dispatch behavior
- Critical: Integer overflow in tensor view bounds check
- Critical: Stack buffer overflow in DAG cycle detection (fixed-size array → dynamic allocation)
- Critical: NULL dereference in scratch size calculation when graph has no nodes
- High: Uninitialized session struct fields (zero-initialized via `memset`)
- High: Missing `pExtents` NULL check in tensor copy region validation
- OOM test for session creation updated to use explicit scratch memory

## [0.1.0] - 2026-03-05

### Added

- Initial reference implementation of VK_KHR_ml_primitives
- 13 Vulkan API entry points
- 6 primitive descriptor types (convolution, GEMM, pooling, activation, normalization, elementwise)
- Validation layer with 60 VUID checks
- Conformance test suite with 13 test executables
- Quickstart example
- CMake build system with C17 support
