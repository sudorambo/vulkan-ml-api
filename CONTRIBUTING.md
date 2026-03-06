# Contributing to Vulkan ML API

Thank you for your interest in contributing to the VK_KHR_ml_primitives reference implementation.

## Development Setup

### Prerequisites

- C17-compatible compiler (GCC 8+, Clang 6+, MSVC 2019+)
- CMake 3.20+
- Vulkan SDK 1.3+

### Building

```bash
cmake --preset default
cmake --build build
ctest --test-dir build --output-on-failure
```

## Code Standards

### Language and Style

- All source code is C17 (`-std=c17`).
- Follow the existing code style (see `.clang-format`).
- All warnings are errors (`-Werror` / `/WX`).
- Run `clang-tidy` before submitting (see `.clang-tidy` config).

### VUID Traceability

Every Vulkan Usage ID (VUID) from the spec must have:

1. A `#define VUID_*` constant in `src/internal.h`
2. A validation check in the appropriate `layers/validation/*.c` file
3. At least one negative test case

When adding a new VUID:

1. Add the `#define` to `src/internal.h`
2. Implement the validation check
3. Add a negative test in `tests/validation/` or `tests/unit/`

### Commit Discipline

- Write clear, descriptive commit messages
- Reference the VUID or spec section when applicable
- One logical change per commit
- Ensure all tests pass before committing

## Architecture

```
src/                    # ICD reference implementation
layers/validation/      # Validation layer (returns VK_TRUE/VK_FALSE)
tests/cts/              # Conformance test suite
tests/unit/             # Unit tests for validation functions
tests/validation/       # VUID-specific validation tests
include/vulkan/         # Public API header
spec/                   # Extension specification
```

### Adding a New Operation Type

1. Add the enum value to `VkMLOperationTypeKHR` in `vulkan_ml_primitives.h`
2. Define the descriptor struct if needed
3. Add validation in `layers/validation/graph_validation.c`
4. Add ICD handling in `src/ml_primitives.c`
5. Add test cases
6. Update `review-findings.md` with any new VUIDs

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes following the standards above
3. Ensure all tests pass: `ctest --test-dir build --output-on-failure`
4. Submit a pull request with a clear description
5. Address any review feedback

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include the VUID reference if reporting a spec compliance issue
- Provide minimal reproduction steps
