# Research: v1.0 Release Readiness

**Plan**: [plan.md](plan.md) | **Date**: 2026-03-05

## R1: sType Value Assignment Strategy

**Question**: What is the correct approach for assigning new `VkStructureType`
values to avoid collisions while staying within the extension's registered range?

**Decision**: Assign the 4 colliding sTypes (dispatch info, tensor binding,
dependency info, tensor copy) to the next sequential values in the extension's
range: 1000559024–1000559027.

**Rationale**: Vulkan extensions receive a contiguous block of 1000 values
starting at `1000000000 + (extensionNumber - 1) * 1000`. Extension number
560 (0-indexed 559) gives range 1000559000–1000559999. Values 1000559000–
1000559023 are already assigned. The next 4 available are 1000559024–27.
Since this is pre-1.0 with no published consumers, reassignment is safe.

**Alternatives considered**:
- Reassign the *new* descriptor sTypes instead: rejected because the new
  descriptors (concat/reshape/transpose/resize) were added more recently
  and are less likely to have been used in external tests.
- Use non-sequential values: rejected because sequential assignment is the
  Vulkan convention and simplifies range accounting.

## R2: Deep Copy Strategy for Pointer-Containing Descriptors

**Question**: `VkMLPrimitiveDescReshapeKHR` has `pOutputDimensions` (pointer
to `uint32_t[]`) and `VkMLPrimitiveDescTransposeKHR` has `pPermutation`
(pointer to `uint32_t[]`). The existing `deep_copy_op_desc()` does a flat
`memcpy`. How should these be handled?

**Decision**: Extend `deep_copy_op_desc()` with post-copy logic that detects
reshape and transpose sTypes and deep-copies their pointer members into
separately allocated arrays, similar to how `deep_copy_tensor_desc()` handles
`pDimensions` and `pStrides`.

**Rationale**: The flat `memcpy` copies the pointer value, which becomes
dangling if the caller frees the original. Since `vkCreateMLGraphKHR`
documents that the `pCreateInfo` and all referenced memory may be freed
after the call returns, deep copy is required for correctness.

**Alternatives considered**:
- Reference counting on the source arrays: rejected because Vulkan's
  create/destroy model expects full ownership transfer at creation time.
- Requiring callers to keep descriptor memory alive: rejected because it
  violates Vulkan conventions where create info is consumed and may be freed.

## R3: Integer Overflow Prevention Pattern

**Question**: What is the idiomatic C pattern for overflow-safe bounds
checking of `uint32_t` addition in `offset + size <= limit`?

**Decision**: Rewrite as subtraction: `size > limit || offset > limit - size`.
This avoids the overflow entirely because `limit - size` is valid when
`size <= limit` (guaranteed by the first check).

**Rationale**: This pattern is used extensively in the Vulkan validation
layers (mesa, lvp) and in the CTS. It adds no branches beyond the original
check and works for all `uint32_t` values including `UINT32_MAX`.

**Alternatives considered**:
- Cast to `uint64_t` for the addition: works but is less idiomatic and
  adds unnecessary widening on 32-bit platforms.
- Use compiler builtins (`__builtin_add_overflow`): not portable to MSVC
  without `#ifdef` guards.

## R4: Stack vs Heap for DFS Color Array

**Question**: The DFS cycle detection uses a stack-allocated `uint8_t color[256]`.
Should this be changed to a heap allocation to support arbitrary `maxMLGraphNodeCount`?

**Decision**: For 1.0, add a guard that rejects graphs with `nodeCount > 256`
in the validation layer. Keep the stack allocation.

**Rationale**: The reference ICD's `VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT` is 256,
which is the ICD's own hard limit. The validation layer using the same limit
is consistent. A guard clause is simpler and avoids introducing a new OOM path
in validation code. Post-1.0, if `maxMLGraphNodeCount` increases, dynamic
allocation can be added.

**Alternatives considered**:
- `malloc`/`free` in the validation function: adds complexity, requires OOM
  handling in a function that returns `VkBool32`, and the allocation is small
  enough that stack is appropriate for the reference implementation's limits.
- VLA (variable-length array): forbidden in C11 strict mode and dangerous
  for large values.

## R5: `(int)` Cast Removal for sType Comparisons

**Question**: Multiple files cast `VkStructureType` to `(int)` before
comparing with enum constants. Is this safe? What's the correct pattern?

**Decision**: Remove the `(int)` cast. Compare directly as `uint32_t` using
the pattern `(uint32_t)pCreateInfo->sType == (uint32_t)VK_STRUCTURE_TYPE_...`.

**Rationale**: `VkStructureType` is defined as `uint32_t` in the Vulkan
headers. The anonymous enum constants in `vulkan_ml_primitives.h` have
values > 1 billion, which fit in `uint32_t` but technically exceed `INT_MAX`
on most platforms (where `int` is 32-bit signed). Casting to `(int)` is
implementation-defined behavior in C. The `(uint32_t)` cast is explicit
and safe for all values in the `VkStructureType` range.

**Alternatives considered**:
- No cast at all: works in practice but some compilers warn about
  signed/unsigned comparison between enum and `uint32_t`.
- Define sTypes as `static const uint32_t` instead of anonymous enum:
  would fix the root cause but is a larger refactor and diverges from
  how the Vulkan SDK headers define extension sTypes.

## R6: Activation Descriptor Validation Rules

**Question**: What validation rules apply to `VkMLPrimitiveDescActivationKHR`
that the current no-op case is missing?

**Decision**: Validate:
1. `sType == VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ACTIVATION_KHR`
2. `activationType` is within the valid enum range (0–5, i.e.
   `<= VK_ML_ACTIVATION_FUNCTION_CLAMP_KHR`)
3. `param0` and `param1` are finite (`isfinite()`)
4. For `LEAKY_RELU`: `param0` (alpha) must be finite and typically in (0, 1)
5. For `CLAMP`: `param0 <= param1` (min <= max)

**Rationale**: The ICD's `vk_ml_validate_primitive_desc` already checks
finiteness for activation params (cases `VK_ML_OPERATION_TYPE_RELU_KHR`
through `SOFTMAX_KHR`). The validation layer should additionally check
the `activationType` enum range and semantic constraints (clamp ordering).

**Alternatives considered**:
- Only validate finiteness (matching ICD): insufficient for the validation
  layer, which should catch more errors than the ICD.
- Validate against device features: activation descriptors don't have a
  fused activation field (they *are* the activation), so `fusedActivations`
  feature doesn't apply here.

## R7: CMake NAMESPACE Impact on Downstream Projects

**Question**: Adding `NAMESPACE vk_ml_primitives::` changes the imported
target name. Does this break anything?

**Decision**: Add the namespace. Update `vk_ml_primitivesConfig.cmake.in`
to document the namespaced target name.

**Rationale**: Pre-1.0, no published consumers exist. Adding the namespace
now (before 1.0 freeze) is the right time. The namespace enables CMake's
target name typo detection (`target_link_libraries(app PRIVATE vk_ml_primtives::vk_ml_primitives)` would fail at configure time instead of silently
passing). This is a CMake best practice documented in the CMake packaging
guide.

**Alternatives considered**:
- Skip namespace: simpler but loses the typo-detection benefit and diverges
  from CMake packaging conventions.

## R8: pkg-config Variable Strategy

**Question**: Should `vk_ml_primitives.pc.in` use `@CMAKE_INSTALL_FULL_LIBDIR@`
or `@CMAKE_INSTALL_LIBDIR@`?

**Decision**: Use `@CMAKE_INSTALL_FULL_LIBDIR@` and
`@CMAKE_INSTALL_FULL_INCLUDEDIR@` for absolute paths.

**Rationale**: The `FULL` variants include the install prefix, producing
correct absolute paths like `/usr/local/lib64` on Fedora. Without `FULL`,
the value is a relative path like `lib64` which must be composed with
`${prefix}`, but `${exec_prefix}` vs `${prefix}` handling varies across
pkg-config implementations. Using absolute paths is simpler and more
reliable.

**Alternatives considered**:
- Use `${exec_prefix}/@CMAKE_INSTALL_LIBDIR@`: works but is more complex
  and requires correct `exec_prefix` handling.
