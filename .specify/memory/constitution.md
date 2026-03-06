<!--
  Sync Impact Report
  ==================
  Version change: N/A (initial) → 1.0.0
  Modified principles: N/A (initial ratification)
  Added sections:
    - Core Principles (7 principles)
    - Technical Constraints
    - Development Workflow
    - Governance
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md         ✅ compatible (no changes needed)
    - .specify/templates/spec-template.md          ✅ compatible (no changes needed)
    - .specify/templates/tasks-template.md         ✅ compatible (no changes needed)
    - .specify/templates/checklist-template.md     ✅ compatible (no changes needed)
    - .specify/templates/agent-file-template.md    ✅ compatible (no changes needed)
  Follow-up TODOs: None
-->

# Vulkan ML API Constitution

## Core Principles

### I. Specification-Driven Development

All implementation artifacts MUST trace directly to the authoritative
extension specification (`spec/VK_KHR_ml_primitives.adoc`). The spec
is the single source of truth for API surface, valid usage rules
(VUIDs), and behavioral semantics.

- Code MUST NOT introduce behavior absent from the specification.
- Every Valid Usage ID (VUID) defined in the spec MUST have a
  corresponding validation check and test case before the feature
  is considered complete.
- When ambiguity exists in the spec, the resolution MUST be captured
  as a spec amendment first, then implemented. Implementation MUST
  NOT precede specification.
- Requirements traceability MUST be maintained: each source file,
  test, or validation rule MUST reference the spec section or VUID
  it implements.

### II. Vulkan C API Conventions (NON-NEGOTIABLE)

All C code MUST conform to established Vulkan API patterns without
exception. Consistency with the Vulkan ecosystem is a hard constraint.

- **Language standard**: C11 minimum. No compiler-specific extensions
  in public headers; implementation files MAY use platform-guarded
  extensions where justified.
- **Naming**: `VK_` prefix for enums and constants, `Vk` prefix for
  type names, `vk` prefix for function names. Extension suffixes
  MUST use `KHR` for this cross-vendor extension.
- **Structure patterns**: Every extensible structure MUST carry
  `sType` and `pNext` members as its first two fields. `sType`
  values MUST be unique and registered.
- **Lifecycle**: All Vulkan objects MUST follow explicit
  create/destroy (or allocate/free) pairs. Destroy functions MUST
  be safe to call with `VK_NULL_HANDLE`.
- **Error reporting**: Functions that can fail MUST return `VkResult`.
  Void-returning functions MUST NOT fail.
- **Memory allocation**: All object creation functions MUST accept a
  `const VkAllocationCallbacks*` parameter for host memory control.

### III. Portability and Cross-Vendor Compliance

The API MUST function identically across desktop, mobile, and embedded
Vulkan implementations. No vendor-specific assumptions are permitted
in the specification or reference implementation.

- Every optional capability MUST be queryable via
  `VkPhysicalDeviceMLFeaturesKHR` or
  `VkPhysicalDeviceMLPropertiesKHR` before use.
- Implementation MUST NOT assume specific hardware topology (e.g.,
  dedicated ML accelerator vs. general compute).
- Data types and alignments MUST use Vulkan-defined types
  (`VkDeviceSize`, `uint32_t`, etc.) rather than platform-specific
  types.
- All numeric limits MUST be exposed as queryable properties, never
  as compile-time constants.
- Endianness-sensitive code MUST be explicitly handled; the spec
  defines element ordering, not byte ordering.

### IV. Test-First with Validation Layers (NON-NEGOTIABLE)

Validation rules and conformance tests MUST be written before or
concurrently with implementation. No feature is complete without
passing validation and conformance suites.

- For each VUID in the spec, a validation layer check MUST exist
  that detects the violation and emits a diagnostic message
  referencing the VUID string.
- Conformance test cases (CTS) MUST exercise every API entry point,
  every enumeration value, and every documented error path.
- Tests MUST be device-independent: they MUST query capabilities
  and skip gracefully on unsupported hardware rather than fail.
- Negative tests MUST verify that invalid usage is detected and
  does not cause undefined behavior, crashes, or silent corruption.
- Static analysis (e.g., `clang-tidy`, `cppcheck`) MUST be run on
  all C source with zero warnings in CI.

### V. Explicit Resource Lifecycle and Memory Safety

All resource and memory management MUST be explicit and auditable.
No hidden allocations, no implicit state transitions, no use-after-
free by design.

- Tensor, graph, session, and view objects MUST follow the Vulkan
  create/bind/use/destroy lifecycle. No object may be used before
  memory binding is complete.
- Memory requirements MUST be queryable before allocation
  (`vkGetTensorMemoryRequirementsKHR`,
  `vkGetMLGraphMemoryRequirementsKHR`).
- Scratch memory auto-allocation MUST be opt-in via a capability
  flag (`mlGraphScratchAutoAllocation`), never implicit.
- All pointer parameters in C APIs MUST document ownership,
  nullability, and lifetime expectations in the spec.
- Buffer overruns MUST be prevented by validating all
  `dimensionCount`, `regionCount`, and array size parameters
  against queried limits before dereferencing.

### VI. Backward Compatibility and Extension Versioning

Published API surface MUST NOT change in ways that break existing
valid applications. Evolution MUST use Vulkan's extension mechanisms.

- New functionality MUST be added via `pNext` structure chaining or
  new extension interactions, never by modifying existing structure
  layouts.
- The extension revision number MUST increment for any behavioral
  change, new enum value, or new function.
- Deprecated features MUST remain functional for at least one full
  revision cycle with documented migration guidance.
- `VkStructureType` values, object type enums, and function pointers
  MUST remain stable once assigned; reassignment is forbidden.
- Semantic versioning applies to the constitution itself (see
  Governance below) and to specification revision tracking.

### VII. Simplicity and Composability

Prefer composing existing Vulkan primitives over introducing novel
abstractions. Every new concept MUST justify its existence against
reuse of existing Vulkan patterns.

- Synchronization MUST reuse `VkPipelineBarrier2` and timeline
  semaphores. No new synchronization primitives are introduced.
- ML graphs record into standard `VkCommandBuffer` objects and
  submit via `vkQueueSubmit2`. No custom submission paths.
- Tensor objects follow the image/buffer lifecycle pattern. No
  novel resource lifecycle is introduced.
- YAGNI: features MUST NOT be added speculatively. Each API
  surface element MUST address a concrete, demonstrable use case
  documented in the spec.
- Complexity MUST be justified: if a simpler alternative achieves
  the same goal, the simpler alternative MUST be chosen.

## Technical Constraints

- **Primary language**: C (C11 standard minimum; C17 preferred for
  implementation files).
- **Vulkan baseline**: Vulkan 1.3 required. Dependencies on
  `VK_KHR_cooperative_matrix`, `VK_KHR_timeline_semaphore`,
  `VK_KHR_maintenance5`, `VK_KHR_format_feature_flags2`, and
  `SPV_KHR_tensor`.
- **Build system**: MUST support CMake 3.20+ as the primary build
  system. Build MUST be reproducible given identical source and
  toolchain.
- **Compiler support**: GCC 11+, Clang 14+, MSVC 2022+. Public
  headers MUST compile cleanly under `-Wall -Wextra -Wpedantic`
  (or equivalent) with zero warnings.
- **Platform targets**: Linux (x86_64, aarch64), Windows (x86_64),
  Android (aarch64), and any Vulkan 1.3 capable embedded target.
- **SPIR-V**: The `SPV_KHR_tensor` extension MUST be co-maintained.
  Shader-accessible tensor operations MUST be expressible in SPIR-V
  without vendor-specific instructions.
- **Thread safety**: All `vkCmd*` recording functions are
  externally synchronized per command buffer (standard Vulkan
  threading model). Object creation/destruction functions MUST be
  safe to call concurrently for independent objects.
- **No dynamic memory in hot paths**: Dispatch (`vkCmdDispatchMLGraphKHR`)
  and barrier operations MUST NOT allocate host memory.

## Development Workflow

- **Spec-first cycle**: Specification amendment → review → approval →
  validation layer update → CTS update → reference implementation
  update. No step may be skipped or reordered.
- **Code review**: All changes MUST be reviewed by at least one
  contributor other than the author before merging. Reviews MUST
  verify VUID traceability, memory safety, and Vulkan convention
  compliance.
- **CI gates**: Every merge request MUST pass: (1) full build on
  all supported compilers, (2) static analysis with zero warnings,
  (3) validation layer test suite, (4) CTS execution on at least
  one conformant implementation or simulator.
- **Branch strategy**: Feature branches off `main`. Direct pushes
  to `main` are forbidden. Release tags follow `vN` format aligned
  with the extension revision number.
- **Commit discipline**: Each commit MUST be atomic (single logical
  change), buildable, and testable independently. Commit messages
  MUST reference the spec section or VUID being addressed.
- **Documentation**: Public API functions, structures, and enums
  MUST have complete Doxygen-style documentation in headers that
  mirrors the spec language. Changes to the API MUST include
  corresponding spec `.adoc` updates in the same merge request.

## Governance

This constitution is the supreme governing document for the Vulkan
ML API project. It supersedes all other conventions, guidelines, or
ad-hoc practices. All contributors, reviewers, and automated
tooling MUST verify compliance with these principles.

- **Amendment procedure**: Any principle or constraint MAY be
  amended by opening a proposal as a merge request against this
  file. Amendments MUST include rationale, impact analysis, and a
  migration plan for existing code. Approval requires review and
  sign-off from at least two project maintainers.
- **Versioning policy**: This constitution follows semantic
  versioning. MAJOR bumps for principle removals or incompatible
  redefinitions. MINOR bumps for new principles or material
  expansions. PATCH bumps for clarifications, typos, or
  non-semantic wording changes.
- **Compliance review**: All merge requests MUST include a
  constitution compliance statement confirming the change adheres
  to the applicable principles. Reviewers MUST flag violations
  before approval.
- **Dispute resolution**: When a proposed change conflicts with a
  principle, the principle takes precedence unless formally amended
  first. "Fix the constitution, then fix the code" — never the
  reverse.
- **Runtime guidance**: Use the project's development guidelines
  file (generated from `.specify/templates/agent-file-template.md`)
  for day-to-day development guidance that supplements but does not
  override this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-03-05 | **Last Amended**: 2026-03-05
