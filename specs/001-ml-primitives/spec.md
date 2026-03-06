# Feature Specification: VK_KHR_ml_primitives

**Feature Branch**: `001-ml-primitives`  
**Created**: 2026-03-05  
**Status**: Draft  
**Input**: Vulkan ML extension specification at `spec/VK_KHR_ml_primitives.adoc`

## User Scenarios & Testing

### User Story 1 - Tensor Resource Management (Priority: P1)

A GPU application developer needs to create, configure, and manage
multi-dimensional tensor data on the GPU for machine learning workloads.
Tensors serve as the fundamental data container — the developer must be
able to specify dimensions (up to 8D), element format (FP16, BF16, INT8,
FP8, etc.), memory tiling strategy, and usage intent. The developer must
also query memory requirements and bind device memory before use.

**Why this priority**: Without tensor resources, no ML operation can
execute. This is the foundational data type that every other feature
depends on.

**Independent Test**: Create tensors of various shapes and formats, bind
memory, verify memory requirement queries return valid results, and
confirm tensors can be destroyed without leaks or errors.

**Acceptance Scenarios**:

1. **Given** a Vulkan device with `tensorObjects` enabled, **When** the
   developer creates a 4D FP16 tensor with shape [1, 3, 224, 224] and
   optimal tiling, **Then** the tensor is created successfully and memory
   requirements are queryable.
2. **Given** a created tensor, **When** the developer allocates compatible
   device memory and binds it, **Then** the tensor is ready for use in
   ML graphs or shader access.
3. **Given** a tensor with memory bound, **When** the developer creates
   a tensor view with a sub-region and compatible format reinterpretation,
   **Then** the view provides access to the specified sub-region without
   copying data.
4. **Given** two tensors with transfer usage flags, **When** the developer
   records a tensor copy command specifying source/destination regions,
   **Then** data is copied between the specified regions correctly.

---

### User Story 2 - ML Graph Construction and Compilation (Priority: P2)

A framework developer needs to compose multiple ML operations (convolutions,
matrix multiplications, pooling, activations, normalization, elementwise
operations) into a directed acyclic graph (DAG) that can be compiled once
and dispatched multiple times. The graph defines external inputs, outputs,
and constant weights, with internal edges connecting intermediate results
between nodes.

**Why this priority**: The ML graph is the primary execution abstraction.
Without it, individual primitives have no execution mechanism. Graph
compilation enables IHV optimization across the full operation DAG.

**Independent Test**: Construct a multi-node graph (e.g., convolution →
batch normalization → ReLU → pooling), verify it compiles successfully,
query scratch memory requirements, and confirm the graph object can be
reused across multiple sessions.

**Acceptance Scenarios**:

1. **Given** a device with `mlPrimitives` and `mlGraph` enabled, **When**
   the developer defines a single-node convolution graph with external
   input, weight, and output tensor descriptions, **Then** the graph
   compiles successfully.
2. **Given** a multi-node graph with convolution, normalization, and
   activation nodes connected via internal tensor edges, **When** the
   developer submits it for compilation, **Then** all internal edges are
   validated for shape/format compatibility and the graph is created.
3. **Given** an invalid graph (e.g., cyclic dependencies or mismatched
   tensor shapes between nodes), **When** compilation is attempted,
   **Then** the operation fails with an appropriate error code and no
   resources are leaked.
4. **Given** a compiled graph, **When** scratch memory requirements are
   queried, **Then** a valid memory requirement is returned that the
   developer can use to allocate scratch memory.

---

### User Story 3 - ML Session and Graph Dispatch (Priority: P3)

An application developer needs to execute a compiled ML graph by creating
a session that manages scratch memory and transient state, binding actual
tensor resources (inputs, outputs, weights) at dispatch time, and recording
the dispatch into a standard Vulkan command buffer for submission to a
compute-capable queue.

**Why this priority**: This is the execution path — the culmination of
tensor creation and graph compilation. Multiple sessions enable batched
and pipelined inference.

**Independent Test**: Create a session for a compiled graph, record an
ML dispatch into a command buffer with bound tensors, submit to a queue,
wait for completion, and verify output tensor contains expected results
from the ML operation.

**Acceptance Scenarios**:

1. **Given** a compiled ML graph and allocated scratch memory, **When**
   the developer creates a session binding the graph and scratch memory,
   **Then** the session is created successfully.
2. **Given** a valid session and tensors matching the graph's external
   descriptions, **When** the developer records
   `vkCmdDispatchMLGraphKHR` into a command buffer and submits to a
   compute queue, **Then** the ML graph executes and produces correct
   output.
3. **Given** auto-allocation support (`mlGraphScratchAutoAllocation`),
   **When** the developer creates a session without explicit scratch
   memory, **Then** the implementation handles scratch allocation
   internally.
4. **Given** multiple sessions for the same graph, **When** dispatches
   are submitted concurrently from different command buffers, **Then**
   each session executes independently without interference.

---

### User Story 4 - Synchronization and Interop with Vulkan Pipelines (Priority: P4)

A developer building a hybrid rendering/inference pipeline needs to
synchronize ML graph dispatches with other Vulkan operations (compute
shaders, graphics rendering) using existing Vulkan synchronization
primitives — pipeline barriers with tensor memory barriers and timeline
semaphores for cross-queue coordination.

**Why this priority**: Real-world applications mix ML inference with
rendering (neural rendering, super-resolution, inference-in-the-loop).
Without synchronization, outputs cannot be safely consumed by downstream
stages.

**Independent Test**: Dispatch an ML graph, insert a tensor memory barrier,
then read the output tensor from a compute shader. Verify the barrier
correctly orders the operations and no data races occur.

**Acceptance Scenarios**:

1. **Given** an ML graph dispatch that writes to an output tensor, **When**
   the developer inserts a pipeline barrier transitioning from ML write
   access to shader read access, **Then** a subsequent compute shader
   can safely read the tensor output.
2. **Given** an ML dispatch on a dedicated ML queue and a compute
   dispatch on a compute queue, **When** a timeline semaphore is used
   to order the operations, **Then** the compute dispatch waits for
   the ML dispatch to complete before reading tensor data.
3. **Given** a tensor shared between queue families with exclusive
   sharing mode, **When** the developer performs a queue family ownership
   transfer via paired tensor memory barriers, **Then** the tensor is
   safely accessible from the destination queue family.

---

### User Story 5 - SPIR-V Tensor Shader Access (Priority: P5)

A developer writing custom compute shaders needs to access tensor data
directly from SPIR-V using typed tensor accessors, enabling hybrid
workloads where custom shader logic operates alongside opaque ML
primitives on the same tensor resources.

**Why this priority**: Not all ML operations are covered by fixed-function
primitives. Shader access enables custom pre/post-processing, novel
activations, and operations the extension does not natively support.

**Independent Test**: Create a tensor, bind it as a descriptor, write a
compute shader that reads/writes tensor elements via SPIR-V tensor
instructions, dispatch the shader, and verify correct element access.

**Acceptance Scenarios**:

1. **Given** a tensor with `VK_TENSOR_USAGE_SHADER_BIT_KHR` and the
   `tensorShaderAccess` feature enabled, **When** the developer binds
   a tensor view to a descriptor set using `VK_DESCRIPTOR_TYPE_TENSOR_KHR`,
   **Then** the tensor is accessible from a compute shader.
2. **Given** a SPIR-V shader using `OpTensorReadKHR` and `OpTensorWriteKHR`
   with the `TensorShaderAccessKHR` capability, **When** the shader is
   dispatched over a tensor, **Then** elements are read and written
   correctly at the specified coordinates.
3. **Given** a tensor with dynamic dimensions, **When** the shader uses
   `OpTensorQuerySizeKHR` to query a dimension at runtime, **Then** the
   correct dimension size is returned.

---

### Edge Cases

- What happens when a tensor is created with zero-size dimensions? The
  system MUST reject this with a validation error — all dimension values
  must be greater than zero.
- How does the system handle exceeding `maxTensorDimensions` or
  `maxTensorElements` limits? The system MUST reject creation with a
  validation error referencing the exceeded limit.
- What happens when a graph node references a non-existent node index
  for an internal tensor edge? Graph compilation MUST fail with an
  appropriate error.
- How are out-of-bounds tensor accesses handled in SPIR-V? When the
  `OutOfBoundsZero` operand is specified, reads return zero and writes
  are discarded. Without it, behavior is undefined per Vulkan conventions.
- What happens when scratch memory is insufficient for a graph dispatch?
  If explicit scratch is provided, the session creation MUST fail if
  the provided size is less than the requirement. If auto-allocation
  is used, the implementation handles sizing internally.
- What happens when `vkCmdDispatchMLGraphKHR` is called inside a render
  pass? This MUST be rejected — ML dispatches are only valid outside
  render pass instances.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST support creation of N-dimensional tensor
  resources with up to 8 dimensions, configurable element format,
  tiling mode (optimal or linear), and usage flags.
- **FR-002**: The system MUST expose memory requirements for tensors
  so applications can allocate and bind compatible device memory
  before use.
- **FR-003**: The system MUST support tensor views that enable format
  reinterpretation and sub-region access without data copies.
- **FR-004**: The system MUST support copying data between tensors
  via command buffer recorded operations with specified source and
  destination regions.
- **FR-005**: The system MUST support at least 21 ML primitive
  operation types: 2D convolution, deconvolution, GEMM, fully
  connected, max/average/global-average pooling, ReLU, sigmoid,
  tanh, leaky ReLU, PReLU, softmax, batch normalization, LRN,
  elementwise add/multiply, concat, reshape, transpose, and resize.
- **FR-006**: The system MUST support composition of ML primitives
  into a directed acyclic graph (DAG) that is compiled once and
  may be dispatched multiple times.
- **FR-007**: The system MUST support fused activation functions
  (ReLU, sigmoid, tanh, leaky ReLU, clamp) on convolution, GEMM,
  normalization, and elementwise operations when the capability
  is advertised.
- **FR-008**: The system MUST support ML sessions that bind a
  compiled graph to scratch memory for execution.
- **FR-009**: The system MUST support dispatching ML graphs via
  standard command buffers submitted to compute-capable queues.
- **FR-010**: The system MUST expose a new pipeline stage and
  access flags for ML graph operations to enable barrier-based
  synchronization with other Vulkan pipeline stages.
- **FR-011**: The system MUST support tensor memory barriers for
  synchronizing tensor access between ML dispatches, compute
  shaders, and transfer operations.
- **FR-012**: The system MUST support SPIR-V tensor accessors for
  reading, writing, and querying tensor dimensions from compute
  shaders when the capability is advertised.
- **FR-013**: The system MUST expose all optional ML capabilities
  via queryable feature and property structures following the
  existing `VkPhysicalDeviceFeatures2` / `VkPhysicalDeviceProperties2`
  pattern.
- **FR-014**: The system MUST report numeric limits (max dimensions,
  max elements, max graph nodes, max graph depth, max sessions,
  alignment requirements, max scratch size) as queryable device
  properties.
- **FR-015**: The system MUST support multiple tensor element
  formats including FP16, BF16, INT8, INT4, and FP8 (E4M3, E5M2)
  when the corresponding capability is advertised.

### Key Entities

- **Tensor (VkTensorKHR)**: N-dimensional data container with format,
  shape, tiling, and usage. Follows image/buffer lifecycle (create →
  query memory → allocate → bind → use → destroy).
- **Tensor View (VkTensorViewKHR)**: Typed sub-region view into a
  tensor for format reinterpretation or windowed access.
- **ML Graph (VkMLGraphKHR)**: Compiled DAG of ML primitive operations.
  Immutable after creation. Defines external inputs, outputs, weights,
  and internal intermediate edges.
- **ML Session (VkMLSessionKHR)**: Execution context binding a graph
  to scratch memory. Manages transient state for dispatch.
- **ML Primitive Descriptors**: Per-operation configuration structures
  (convolution parameters, GEMM coefficients, pooling window, activation
  type, normalization epsilon, etc.).
- **Tensor Memory Barrier**: Synchronization primitive for ordering
  tensor access between pipeline stages and queue families.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Developers can create tensor resources and execute ML
  graph dispatches using fewer than 10 API calls for a minimal
  single-operation pipeline (create tensor → create graph → create
  session → dispatch).
- **SC-002**: All 21 ML primitive operation types can be composed
  into graphs and dispatched successfully on any conformant
  implementation.
- **SC-003**: The extension works identically across desktop (Linux,
  Windows) and mobile (Android) Vulkan 1.3 implementations without
  application-side platform-specific code.
- **SC-004**: A compiled ML graph can be reused for at least 1000
  dispatches without performance degradation, demonstrating the
  compile-once-dispatch-many model.
- **SC-005**: Tensor data produced by ML graph dispatches can be
  consumed by compute shaders within the same command buffer using
  standard pipeline barriers, with zero data corruption across
  1 million dispatch-barrier-shader cycles.
- **SC-006**: ML graph compilation for a 50-node graph completes
  within a time acceptable for application initialization (not
  per-frame), demonstrating that compilation is a setup-time cost.
- **SC-007**: All valid usage rules (VUIDs) defined in the
  specification have corresponding validation layer checks that
  detect violations and emit diagnostics without false positives.
- **SC-008**: Hybrid workloads (ML dispatch + compute shader on
  same tensors) execute correctly using existing Vulkan
  synchronization primitives with no new synchronization concepts
  required.
