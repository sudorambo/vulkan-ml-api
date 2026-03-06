# Developer Guide

This guide explains the architecture of the VK_KHR_ml_primitives reference implementation and how to extend it.

## Architecture Overview

The project has three layers:

```
┌─────────────────────────────────────────────┐
│  Application (quickstart.c, tests)          │
├─────────────────────────────────────────────┤
│  Validation Layer (layers/validation/)      │
│  Checks VUIDs, returns VK_TRUE/VK_FALSE     │
├─────────────────────────────────────────────┤
│  ICD Reference Implementation (src/)        │
│  Host-side, returns VkResult                │
├─────────────────────────────────────────────┤
│  Vulkan Headers (include/vulkan/)           │
│  Public API types, enums, function decls    │
└─────────────────────────────────────────────┘
```

### ICD Layer (`src/`)

The Installable Client Driver is a host-side reference implementation. It manages object lifecycle (create/destroy), memory binding, graph compilation, and session management. Command buffer operations (`vkCmdCopyTensorKHR`, `vkCmdDispatchMLGraphKHR`) are stubs — actual GPU execution is IHV-specific.

Key files:
- `internal.h` — Shared types (`VkTensorKHR_T`, `VkMLGraphKHR_T`, etc.), VUID constants, helper functions
- `tensor.c` — Tensor create/destroy/bind/memory requirements
- `ml_graph.c` — Graph create/destroy, deep-copy, memory requirements, primitive desc validation
- `ml_session.c` — Session create/destroy, scratch auto-allocation
- `ml_primitives.c` — `vk_ml_validate_primitive_desc()` for all 21 operation types

### Validation Layer (`layers/validation/`)

Separate from the ICD, the validation layer provides development-time checks. Each function returns `VK_TRUE` (valid) or `VK_FALSE` (invalid). The layer never allocates or modifies state.

Key files:
- `tensor_validation.c` — Tensor create, view, bind, copy validation
- `graph_validation.c` — Graph structure, DAG cycle detection, per-node descriptor validation
- `session_validation.c` — Session scratch size, alignment, graph handle validation
- `dispatch_validation.c` — Dispatch parameter validation
- `barrier_validation.c` — Tensor memory barrier validation

## VUID Traceability

Every Valid Usage ID from the spec follows this path:

1. **Spec** (`spec/VK_KHR_ml_primitives.adoc`) — Defines the VUID string
2. **Define** (`src/internal.h`) — `#define VUID_*` constant
3. **Check** (`layers/validation/*.c`) — Validation function with `/* VUID_* */` comment
4. **Test** (`tests/`) — Negative test that triggers the check

### Adding a New VUID

```c
// 1. In src/internal.h
#define VUID_MY_NEW_CHECK \
    "VUID-VkSomeStruct-field-00001"

// 2. In layers/validation/some_validation.c
/* VUID_MY_NEW_CHECK */
if (some_condition_violated)
    return VK_FALSE;

// 3. In tests/validation/test_vuids.c or tests/unit/
static void test_my_new_check(void) {
    // Set up struct with violated condition
    VkBool32 r = vk_ml_validate_something(&invalid_input);
    expect("test_my_new_check", r, VK_FALSE);
}
```

## Adding a New Operation Type

1. **Add enum value** in `include/vulkan/vulkan_ml_primitives.h`:
   ```c
   VK_ML_OPERATION_TYPE_MY_OP_KHR = N,
   ```

2. **Define descriptor struct** (if the operation has parameters):
   ```c
   typedef struct VkMLPrimitiveDescMyOpKHR {
       VkStructureType sType;
       const void*     pNext;
       // operation-specific fields
   } VkMLPrimitiveDescMyOpKHR;
   ```

3. **Add ICD validation** in `src/ml_primitives.c`:
   ```c
   case VK_ML_OPERATION_TYPE_MY_OP_KHR: {
       const VkMLPrimitiveDescMyOpKHR *d = pDesc;
       // validate fields
       break;
   }
   ```

4. **Add validation layer check** in `layers/validation/graph_validation.c`:
   - Add a `vk_ml_validate_my_op_desc()` function
   - Wire it into `vk_ml_validate_graph_create()`'s per-node switch

5. **Add tests**:
   - Positive test in `tests/unit/test_descriptor_validation.c`
   - Negative tests for each invalid parameter
   - CTS graph test in `tests/cts/test_ml_graph.c`

## Build System

### CMake Targets

| Target | Type | Description |
|--------|------|-------------|
| `vk_ml_primitives` | STATIC | ICD reference implementation |
| `vk_ml_validation` | STATIC | Validation layer |
| `quickstart` | EXE | Example program |
| `test_*` | EXE | Test executables |

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTING` | `ON` | Build test executables |
| `BUILD_EXAMPLES` | `ON` | Build example programs |

### CMake Presets

```bash
cmake --preset default    # Debug build with tests
cmake --preset release    # Release build with tests
cmake --preset ci         # CI build (release + compile commands)
```

### Package Installation

```bash
cmake --install build --prefix /usr/local
```

Downstream projects can then use:
```cmake
find_package(vk_ml_primitives REQUIRED)
target_link_libraries(myapp PRIVATE vk_ml_primitives)
```

Or via pkg-config:
```bash
pkg-config --cflags --libs vk_ml_primitives
```

## Testing

### Test Categories

- **CTS** (`tests/cts/`) — Conformance tests exercising the ICD through the public API
- **Validation** (`tests/validation/`) — VUID-specific negative tests
- **Unit** (`tests/unit/`) — Validation function unit tests

### Running Tests

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

### Test Conventions

- Each test function is `static void test_*(void)`
- Use `expect(name, got, want)` for VkBool32 assertions
- Use `expect_vk(name, got, want)` for VkResult assertions
- Forward-declare test functions at the top of the file
- Register tests in `main()` and print summary at the end
