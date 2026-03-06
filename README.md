# VK\_KHR\_ml\_primitives

A cross-vendor Vulkan extension for GPU-accelerated machine learning.

## Overview

`VK_KHR_ml_primitives` introduces native ML execution capabilities into the
Vulkan ecosystem. The extension defines four new object types — `VkTensorKHR`
(N-dimensional data containers), `VkTensorViewKHR` (typed sub-region views),
`VkMLGraphKHR` (compiled DAGs of ML operations), and `VkMLSessionKHR`
(execution contexts) — along with 21 IHV-optimized ML primitive operations
covering convolution, matrix multiplication, pooling, normalization, and
activation functions. Graphs compile into efficient command sequences that
dispatch through standard `VkCommandBuffer` and `vkQueueSubmit2` paths.
Requires Vulkan 1.3.

## Extension Dependencies

- `VK_KHR_cooperative_matrix`
- `VK_KHR_timeline_semaphore` (core in Vulkan 1.2)
- `VK_KHR_maintenance5`
- `VK_KHR_format_feature_flags2`
- `SPV_KHR_tensor` (SPIR-V extension)

## Repository Structure

```
include/
└── vulkan/
    └── vulkan_ml_primitives.h          Public C header (types, enums, prototypes)

src/
├── internal.h                          Internal shared declarations
├── tensor.c                            VkTensorKHR create/destroy/memory
├── tensor_view.c                       VkTensorViewKHR create/destroy
├── tensor_copy.c                       vkCmdCopyTensorKHR
├── ml_primitives.c                     Primitive descriptor validation/setup
├── ml_graph.c                          VkMLGraphKHR create/destroy/memory query
├── ml_session.c                        VkMLSessionKHR create/destroy
├── ml_dispatch.c                       vkCmdDispatchMLGraphKHR
└── feature_query.c                     Feature/property query entry points

layers/
└── validation/
    ├── vk_ml_validation.h              Validation layer shared header
    ├── tensor_validation.c             VUID checks for tensor operations
    ├── graph_validation.c              VUID checks for graph operations
    ├── session_validation.c            VUID checks for session operations
    ├── dispatch_validation.c           VUID checks for dispatch operations
    └── barrier_validation.c            Tensor memory barrier validation

tests/
├── cts/                                Conformance test suites (10 suites)
│   ├── test_tensor_lifecycle.c         Tensor create/bind/destroy
│   ├── test_tensor_view.c             Tensor view creation and access
│   ├── test_tensor_copy.c             Tensor copy operations
│   ├── test_tensor_formats.c          Format support and feature queries
│   ├── test_ml_graph.c                Graph construction and compilation
│   ├── test_ml_session.c              Session create/destroy
│   ├── test_ml_dispatch.c             Graph dispatch execution
│   ├── test_synchronization.c         Barriers and semaphore interop
│   ├── test_spirv_tensor.c            SPIR-V tensor shader access
│   └── test_oom.c                     Allocation failure (OOM) paths
├── validation/
│   └── test_vuids.c                    Negative tests for all VUIDs
└── unit/
    ├── test_dag_validation.c           DAG cycle detection and shape checks
    ├── test_descriptor_validation.c    Primitive descriptor parameter checks
    └── test_validation_coverage.c      Comprehensive validation coverage

examples/
└── quickstart.c                        Minimal end-to-end API example

spec/
└── VK_KHR_ml_primitives.adoc          Authoritative extension specification

specs/001-ml-primitives/                Design artifacts (plan, tasks, findings)

cmake/
├── vk_ml_primitivesConfig.cmake.in     CMake package config template
└── vk_ml_primitives.pc.in             pkg-config template

docs/
└── developer-guide.md                  Architecture and extension guide

.github/workflows/ci.yml               GitHub Actions CI pipeline
CMakeLists.txt                          Root build configuration
CMakePresets.json                       Build presets (default, release, CI)
Doxyfile                                Doxygen API docs configuration
```

## Prerequisites

- **CMake** 3.20+
- **Vulkan SDK** 1.3+ (headers and loader)
- **C compiler** with C17 support:
  - GCC 12+
  - Clang 15+
  - MSVC 2022+

## Building

```sh
cmake --preset default
cmake --build build
```

Or manually:

```sh
cmake -B build -S . -DBUILD_TESTING=ON
cmake --build build
```

The build produces two static libraries:

- `libvk_ml_primitives.a` — reference ICD layer implementation
- `libvk_ml_validation.a` — validation layer

## Running Tests

```sh
cd build && ctest --output-on-failure
```

The test suite includes 14 executables:

| Suite | Type | Coverage |
|-------|------|----------|
| test\_tensor\_lifecycle | CTS | Tensor create, bind, destroy, memory requirements |
| test\_tensor\_view | CTS | Tensor view creation and typed access |
| test\_tensor\_copy | CTS | Tensor-to-tensor copy operations |
| test\_tensor\_formats | CTS | Format support queries and feature enumeration |
| test\_ml\_graph | CTS | Graph construction, compilation, DAG validation |
| test\_ml\_session | CTS | Session lifecycle management |
| test\_ml\_dispatch | CTS | Graph dispatch execution and validation |
| test\_synchronization | CTS | Tensor memory barriers and semaphore interop |
| test\_spirv\_tensor | CTS | SPIR-V tensor read/write/query operations |
| test\_vuids | Validation | Negative tests for all Valid Usage IDs |
| test\_dag\_validation | Unit | DAG cycle detection, topological ordering |
| test\_oom | CTS | OOM (allocation failure) paths for all create functions |
| test\_descriptor\_validation | Unit | ML primitive descriptor parameter checks |
| test\_validation\_coverage | Unit | Tensor view, copy, session, dispatch, graph, boundary, barrier tests |

## Quick Start

See [`examples/quickstart.c`](examples/quickstart.c) for a minimal end-to-end
example demonstrating the complete workflow:

1. Create tensors with `vkCreateTensorKHR`
2. Build an ML graph with `vkCreateMLGraphKHR`
3. Create an execution session with `vkCreateMLSessionKHR`
4. Dispatch the graph with `vkCmdDispatchMLGraphKHR`
5. Clean up all resources in reverse order

## Validation Layer

The `vk_ml_validation` library provides development-time parameter validation
for all ML primitive operations. Every check maps to a Valid Usage ID (VUID)
from the specification.

### Using the Validation Layer

Link against `vk_ml_validation` in your CMake project:

```cmake
find_package(vk_ml_primitives REQUIRED)
target_link_libraries(myapp PRIVATE vk_ml_primitives vk_ml_validation)
```

Call validation functions before or alongside ICD calls:

```c
#include "vk_ml_validation.h"

VkBool32 valid = vk_ml_validate_tensor_create(&createInfo, &features, &props);
if (valid == VK_FALSE) {
    // Handle validation failure — check parameters against VUIDs
}
```

### Coverage

The validation layer provides checks for all 81 VUIDs defined in the spec,
covering tensor lifecycle, graph creation, session management, dispatch
operations, copy commands, and memory barriers.

## Installation

```sh
cmake --install build --prefix /usr/local
```

Downstream CMake projects:

```cmake
find_package(vk_ml_primitives REQUIRED)
target_link_libraries(myapp PRIVATE vk_ml_primitives)
```

pkg-config:

```sh
pkg-config --cflags --libs vk_ml_primitives
```

## API Documentation

Generate API reference documentation with Doxygen:

```sh
doxygen Doxyfile
# Output in docs/html/index.html
```

See [`docs/developer-guide.md`](docs/developer-guide.md) for architecture
details and how to extend the implementation.

## Static Analysis

`clang-tidy` is automatically integrated when detected by CMake. All source
files are checked with `-Wall -Wextra -Wpedantic -Werror`.

If `cppcheck` is installed, an additional analysis target is available:

```sh
cmake --build build --target cppcheck
```

## Specification

The authoritative extension specification lives at
[`spec/VK_KHR_ml_primitives.adoc`](spec/VK_KHR_ml_primitives.adoc).

- **Extension revision**: 1
- **Extension type**: Cross-vendor (KHR)
- **Required Vulkan version**: 1.3
- **SPIR-V integration**: `SPV_KHR_tensor` with `OpTensorReadKHR`,
  `OpTensorWriteKHR`, `OpTensorQuerySizeKHR`

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the
full license text.
