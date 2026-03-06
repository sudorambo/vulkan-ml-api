# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

- ICD `vkCreateMLGraphKHR` now validates per-node primitive descriptors
- ICD `vkCreateMLSessionKHR` implements scratch auto-allocation when `scratchMemory == VK_NULL_HANDLE`
- Graph validation layer now delegates to per-descriptor validators
- Conv group count TODO resolved with deferred-to-dispatch note

### Fixed

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
