/**
 * @file test_validation_coverage.c
 * @brief Comprehensive validation coverage tests for Phase 3 (Shipping Readiness).
 *
 * Tests cover: tensor view, tensor copy, session, dispatch, graph per-node
 * descriptor validation, boundary conditions, and barrier integration.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "vk_ml_validation.h"
#include "internal.h"
#include <stdio.h>
#include <string.h>

static int passed;
static int failed;

static void expect(const char *name, VkBool32 got, VkBool32 want)
{
    if (got == want) {
        (void)printf("PASS: %s\n", name);
        passed++;
    } else {
        (void)printf("FAIL: %s (got %s, want %s)\n",
                     name,
                     got ? "VK_TRUE" : "VK_FALSE",
                     want ? "VK_TRUE" : "VK_FALSE");
        failed++;
    }
}

static void expect_vk(const char *name, VkResult got, VkResult want)
{
    if (got == want) {
        (void)printf("PASS: %s\n", name);
        passed++;
    } else {
        (void)printf("FAIL: %s (got %d, want %d)\n", name, got, want);
        failed++;
    }
}

/* ------------------------------------------------------------------ */
/* Helper: create a minimal valid tensor for view tests               */
/* ------------------------------------------------------------------ */

static VkTensorKHR_T make_bound_tensor(void)
{
    static const uint32_t dims[] = {4, 8};
    VkTensorKHR_T t;
    memset(&t, 0, sizeof(t));
    t.description.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    t.description.format = VK_FORMAT_R32_SFLOAT;
    t.description.dimensionCount = 2;
    t.description.pDimensions = dims;
    t.memoryBound = VK_TRUE;
    t.dimensions = (uint32_t *)dims;
    return t;
}

static VkPhysicalDeviceMLFeaturesKHR make_features(void)
{
    VkPhysicalDeviceMLFeaturesKHR f;
    memset(&f, 0, sizeof(f));
    f.sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR;
    vk_ml_populate_features(&f);
    return f;
}

static VkPhysicalDeviceMLPropertiesKHR make_props(void)
{
    VkPhysicalDeviceMLPropertiesKHR p;
    memset(&p, 0, sizeof(p));
    p.sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR;
    vk_ml_populate_properties(&p);
    return p;
}

/* ================================================================== */
/* Tensor View Validation Tests                                       */
/* ================================================================== */

static void test_tensor_view_valid(void)
{
    VkTensorKHR_T tensor = make_bound_tensor();
    uint32_t offsets[] = {0, 0};
    uint32_t sizes[] = {4, 8};
    VkTensorViewCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensionOffsets = offsets,
        .pDimensionSizes = sizes,
    };
    VkBool32 r = vk_ml_validate_tensor_view_create(&ci, &tensor);
    expect("tensor_view_valid", r, VK_TRUE);
}

static void test_tensor_view_unbound_memory(void)
{
    VkTensorKHR_T tensor = make_bound_tensor();
    tensor.memoryBound = VK_FALSE;
    uint32_t offsets[] = {0, 0};
    uint32_t sizes[] = {4, 8};
    VkTensorViewCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensionOffsets = offsets,
        .pDimensionSizes = sizes,
    };
    VkBool32 r = vk_ml_validate_tensor_view_create(&ci, &tensor);
    expect("tensor_view_unbound_memory", r, VK_FALSE);
}

static void test_tensor_view_format_mismatch(void)
{
    VkTensorKHR_T tensor = make_bound_tensor();
    uint32_t offsets[] = {0, 0};
    uint32_t sizes[] = {4, 8};
    VkTensorViewCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 2,
        .pDimensionOffsets = offsets,
        .pDimensionSizes = sizes,
    };
    VkBool32 r = vk_ml_validate_tensor_view_create(&ci, &tensor);
    expect("tensor_view_format_mismatch", r, VK_FALSE);
}

static void test_tensor_view_dim_count_mismatch(void)
{
    VkTensorKHR_T tensor = make_bound_tensor();
    uint32_t offsets[] = {0, 0, 0};
    uint32_t sizes[] = {4, 8, 1};
    VkTensorViewCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 3,
        .pDimensionOffsets = offsets,
        .pDimensionSizes = sizes,
    };
    VkBool32 r = vk_ml_validate_tensor_view_create(&ci, &tensor);
    expect("tensor_view_dim_count_mismatch", r, VK_FALSE);
}

static void test_tensor_view_range_overflow(void)
{
    VkTensorKHR_T tensor = make_bound_tensor();
    uint32_t offsets[] = {2, 5};
    uint32_t sizes[] = {4, 8}; /* offset+size = 6,13 > 4,8 */
    VkTensorViewCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensionOffsets = offsets,
        .pDimensionSizes = sizes,
    };
    VkBool32 r = vk_ml_validate_tensor_view_create(&ci, &tensor);
    expect("tensor_view_range_overflow", r, VK_FALSE);
}

static void test_tensor_view_null_create_info(void)
{
    VkTensorKHR_T tensor = make_bound_tensor();
    VkBool32 r = vk_ml_validate_tensor_view_create(NULL, &tensor);
    expect("tensor_view_null_create_info", r, VK_FALSE);
}

/* ================================================================== */
/* Tensor Copy Validation Tests                                       */
/* ================================================================== */

static void test_tensor_copy_valid(void)
{
    uint32_t srcOff[] = {0, 0};
    uint32_t dstOff[] = {0, 0};
    uint32_t sizes[] = {4, 4};
    VkTensorCopyKHR region = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_COPY_KHR,
        .pNext = NULL,
        .dimensionCount = 2,
        .pSrcOffsets = srcOff,
        .pDstOffsets = dstOff,
        .pExtents = sizes,
    };
    VkCopyTensorInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR,
        .pNext = NULL,
        .srcTensor = (VkTensorKHR)(uintptr_t)0x1,
        .dstTensor = (VkTensorKHR)(uintptr_t)0x2,
        .regionCount = 1,
        .pRegions = &region,
    };
    VkBool32 r = vk_ml_validate_tensor_copy(&ci);
    expect("tensor_copy_valid", r, VK_TRUE);
}

static void test_tensor_copy_same_tensor(void)
{
    VkTensorCopyKHR region = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_COPY_KHR,
        .pNext = NULL,
        .dimensionCount = 0,
    };
    VkCopyTensorInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR,
        .pNext = NULL,
        .srcTensor = (VkTensorKHR)(uintptr_t)0x1,
        .dstTensor = (VkTensorKHR)(uintptr_t)0x1,
        .regionCount = 1,
        .pRegions = &region,
    };
    VkBool32 r = vk_ml_validate_tensor_copy(&ci);
    expect("tensor_copy_same_tensor", r, VK_FALSE);
}

static void test_tensor_copy_zero_regions(void)
{
    VkCopyTensorInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR,
        .pNext = NULL,
        .srcTensor = (VkTensorKHR)(uintptr_t)0x1,
        .dstTensor = (VkTensorKHR)(uintptr_t)0x2,
        .regionCount = 0,
        .pRegions = NULL,
    };
    VkBool32 r = vk_ml_validate_tensor_copy(&ci);
    expect("tensor_copy_zero_regions", r, VK_FALSE);
}

static void test_tensor_copy_null_regions_ptr(void)
{
    VkCopyTensorInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR,
        .pNext = NULL,
        .srcTensor = (VkTensorKHR)(uintptr_t)0x1,
        .dstTensor = (VkTensorKHR)(uintptr_t)0x2,
        .regionCount = 1,
        .pRegions = NULL,
    };
    VkBool32 r = vk_ml_validate_tensor_copy(&ci);
    expect("tensor_copy_null_regions_ptr", r, VK_FALSE);
}

static void test_tensor_copy_null_src_offsets(void)
{
    VkTensorCopyKHR region = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_COPY_KHR,
        .pNext = NULL,
        .dimensionCount = 2,
        .pSrcOffsets = NULL,
        .pDstOffsets = NULL,
    };
    VkCopyTensorInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_KHR,
        .pNext = NULL,
        .srcTensor = (VkTensorKHR)(uintptr_t)0x1,
        .dstTensor = (VkTensorKHR)(uintptr_t)0x2,
        .regionCount = 1,
        .pRegions = &region,
    };
    VkBool32 r = vk_ml_validate_tensor_copy(&ci);
    expect("tensor_copy_null_src_offsets", r, VK_FALSE);
}

static void test_tensor_copy_null_info(void)
{
    VkBool32 r = vk_ml_validate_tensor_copy(NULL);
    expect("tensor_copy_null_info", r, VK_FALSE);
}

/* ================================================================== */
/* Session Validation Tests                                           */
/* ================================================================== */

static void test_session_valid_explicit_scratch(void)
{
    VkMLSessionCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = (VkMLGraphKHR)(uintptr_t)0x1,
        .scratchMemory = (VkDeviceMemory)(uintptr_t)0x1,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = 1024,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkBool32 r = vk_ml_validate_session_create(&ci, 512, &features);
    expect("session_valid_explicit_scratch", r, VK_TRUE);
}

static void test_session_null_graph(void)
{
    VkMLSessionCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = VK_NULL_HANDLE,
        .scratchMemory = (VkDeviceMemory)(uintptr_t)0x1,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = 1024,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkBool32 r = vk_ml_validate_session_create(&ci, 512, &features);
    expect("session_null_graph", r, VK_FALSE);
}

static void test_session_insufficient_scratch(void)
{
    VkMLSessionCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = (VkMLGraphKHR)(uintptr_t)0x1,
        .scratchMemory = (VkDeviceMemory)(uintptr_t)0x1,
        .scratchMemoryOffset = 0,
        .scratchMemorySize = 64,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkBool32 r = vk_ml_validate_session_create(&ci, 512, &features);
    expect("session_insufficient_scratch", r, VK_FALSE);
}

static void test_session_auto_alloc_without_feature(void)
{
    VkMLSessionCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = (VkMLGraphKHR)(uintptr_t)0x1,
        .scratchMemory = VK_NULL_HANDLE,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    features.mlGraphScratchAutoAllocation = VK_FALSE;
    VkBool32 r = vk_ml_validate_session_create(&ci, 512, &features);
    expect("session_auto_alloc_without_feature", r, VK_FALSE);
}

static void test_session_auto_alloc_with_feature(void)
{
    VkMLSessionCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_SESSION_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .graph = (VkMLGraphKHR)(uintptr_t)0x1,
        .scratchMemory = VK_NULL_HANDLE,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    features.mlGraphScratchAutoAllocation = VK_TRUE;
    VkBool32 r = vk_ml_validate_session_create(&ci, 512, &features);
    expect("session_auto_alloc_with_feature", r, VK_TRUE);
}

static void test_session_null_info(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkBool32 r = vk_ml_validate_session_create(NULL, 0, &features);
    expect("session_null_info", r, VK_FALSE);
}

/* ================================================================== */
/* Dispatch Validation Tests                                          */
/* ================================================================== */

static void test_dispatch_valid(void)
{
    VkTensorKHR inputs[] = { (VkTensorKHR)(uintptr_t)0x1 };
    VkTensorKHR outputs[] = { (VkTensorKHR)(uintptr_t)0x2 };
    VkMLGraphDispatchInfoKHR di = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = inputs,
        .outputTensorCount = 1,
        .pOutputTensors = outputs,
        .weightTensorCount = 0,
        .pWeightTensors = NULL,
    };
    VkBool32 r = vk_ml_validate_dispatch(&di);
    expect("dispatch_valid", r, VK_TRUE);
}

static void test_dispatch_null_session(void)
{
    VkTensorKHR inputs[] = { (VkTensorKHR)(uintptr_t)0x1 };
    VkTensorKHR outputs[] = { (VkTensorKHR)(uintptr_t)0x2 };
    VkMLGraphDispatchInfoKHR di = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = VK_NULL_HANDLE,
        .inputTensorCount = 1,
        .pInputTensors = inputs,
        .outputTensorCount = 1,
        .pOutputTensors = outputs,
    };
    VkBool32 r = vk_ml_validate_dispatch(&di);
    expect("dispatch_null_session", r, VK_FALSE);
}

static void test_dispatch_zero_inputs(void)
{
    VkTensorKHR outputs[] = { (VkTensorKHR)(uintptr_t)0x2 };
    VkMLGraphDispatchInfoKHR di = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 0,
        .pInputTensors = NULL,
        .outputTensorCount = 1,
        .pOutputTensors = outputs,
    };
    VkBool32 r = vk_ml_validate_dispatch(&di);
    expect("dispatch_zero_inputs", r, VK_FALSE);
}

static void test_dispatch_zero_outputs(void)
{
    VkTensorKHR inputs[] = { (VkTensorKHR)(uintptr_t)0x1 };
    VkMLGraphDispatchInfoKHR di = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = inputs,
        .outputTensorCount = 0,
        .pOutputTensors = NULL,
    };
    VkBool32 r = vk_ml_validate_dispatch(&di);
    expect("dispatch_zero_outputs", r, VK_FALSE);
}

static void test_dispatch_null_input_ptr(void)
{
    VkTensorKHR outputs[] = { (VkTensorKHR)(uintptr_t)0x2 };
    VkMLGraphDispatchInfoKHR di = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = NULL,
        .outputTensorCount = 1,
        .pOutputTensors = outputs,
    };
    VkBool32 r = vk_ml_validate_dispatch(&di);
    expect("dispatch_null_input_ptr", r, VK_FALSE);
}

static void test_dispatch_null_output_ptr(void)
{
    VkTensorKHR inputs[] = { (VkTensorKHR)(uintptr_t)0x1 };
    VkMLGraphDispatchInfoKHR di = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = inputs,
        .outputTensorCount = 1,
        .pOutputTensors = NULL,
    };
    VkBool32 r = vk_ml_validate_dispatch(&di);
    expect("dispatch_null_output_ptr", r, VK_FALSE);
}

static void test_dispatch_weights_nonzero_null_ptr(void)
{
    VkTensorKHR inputs[] = { (VkTensorKHR)(uintptr_t)0x1 };
    VkTensorKHR outputs[] = { (VkTensorKHR)(uintptr_t)0x2 };
    VkMLGraphDispatchInfoKHR di = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_DISPATCH_INFO_KHR,
        .pNext = NULL,
        .session = (VkMLSessionKHR)(uintptr_t)0x1,
        .inputTensorCount = 1,
        .pInputTensors = inputs,
        .outputTensorCount = 1,
        .pOutputTensors = outputs,
        .weightTensorCount = 2,
        .pWeightTensors = NULL,
    };
    VkBool32 r = vk_ml_validate_dispatch(&di);
    expect("dispatch_weights_nonzero_null_ptr", r, VK_FALSE);
}

static void test_dispatch_null_info(void)
{
    VkBool32 r = vk_ml_validate_dispatch(NULL);
    expect("dispatch_null_info", r, VK_FALSE);
}

/* ================================================================== */
/* Graph Per-Node Descriptor Validation Tests                         */
/* ================================================================== */

static VkMLGraphCreateInfoKHR make_single_node_graph(
    VkMLGraphNodeCreateInfoKHR *node,
    VkMLTensorBindingKHR *input,
    VkMLTensorBindingKHR *output,
    const VkTensorDescriptionKHR *inDesc,
    const VkTensorDescriptionKHR *outDesc)
{
    memset(input, 0, sizeof(*input));
    input->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    input->bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR;
    input->pTensorDescription = inDesc;

    memset(output, 0, sizeof(*output));
    output->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    output->bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR;
    output->pTensorDescription = outDesc;

    memset(node, 0, sizeof(*node));
    node->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR;
    node->inputCount = 1;
    node->pInputs = input;
    node->outputCount = 1;
    node->pOutputs = output;

    VkMLGraphCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR;
    ci.nodeCount = 1;
    ci.pNodes = node;
    ci.externalInputCount = 1;
    ci.pExternalInputDescriptions = inDesc;
    ci.externalOutputCount = 1;
    ci.pExternalOutputDescriptions = outDesc;
    return ci;
}

static void test_graph_rejects_invalid_conv_node(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();

    VkMLPrimitiveDescConvolutionKHR conv;
    memset(&conv, 0, sizeof(conv));
    conv.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR;
    conv.strideX = 0; /* invalid */
    conv.strideY = 1;
    conv.dilationX = 1;
    conv.dilationY = 1;
    conv.kernelWidth = 3;
    conv.kernelHeight = 3;
    conv.groupCount = 1;
    conv.paddingMode = VK_ML_PADDING_MODE_VALID_KHR;

    static const uint32_t dims[] = {1, 1, 4, 4};
    VkTensorDescriptionKHR desc;
    memset(&desc, 0, sizeof(desc));
    desc.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.dimensionCount = 4;
    desc.pDimensions = dims;

    VkMLGraphNodeCreateInfoKHR node;
    VkMLTensorBindingKHR input, output;
    VkMLGraphCreateInfoKHR ci = make_single_node_graph(&node, &input, &output, &desc, &desc);
    node.operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR;
    node.pOperationDesc = &conv;

    VkBool32 r = vk_ml_validate_graph_create(&ci, &features, &props);
    expect("graph_rejects_invalid_conv_node", r, VK_FALSE);
}

static void test_graph_accepts_valid_conv_node(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();

    VkMLPrimitiveDescConvolutionKHR conv;
    memset(&conv, 0, sizeof(conv));
    conv.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR;
    conv.strideX = 1;
    conv.strideY = 1;
    conv.dilationX = 1;
    conv.dilationY = 1;
    conv.kernelWidth = 3;
    conv.kernelHeight = 3;
    conv.groupCount = 1;
    conv.paddingMode = VK_ML_PADDING_MODE_VALID_KHR;

    static const uint32_t dims[] = {1, 1, 4, 4};
    VkTensorDescriptionKHR desc;
    memset(&desc, 0, sizeof(desc));
    desc.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.dimensionCount = 4;
    desc.pDimensions = dims;

    VkMLGraphNodeCreateInfoKHR node;
    VkMLTensorBindingKHR input, output;
    VkMLGraphCreateInfoKHR ci = make_single_node_graph(&node, &input, &output, &desc, &desc);
    node.operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR;
    node.pOperationDesc = &conv;

    VkBool32 r = vk_ml_validate_graph_create(&ci, &features, &props);
    expect("graph_accepts_valid_conv_node", r, VK_TRUE);
}

static void test_graph_rejects_invalid_pooling_node(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();

    VkMLPrimitiveDescPoolingKHR pool;
    memset(&pool, 0, sizeof(pool));
    pool.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR;
    pool.poolType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR;
    pool.windowWidth = 0; /* invalid for non-global pool */
    pool.windowHeight = 2;
    pool.strideX = 1;
    pool.strideY = 1;

    static const uint32_t dims[] = {1, 1, 4, 4};
    VkTensorDescriptionKHR desc;
    memset(&desc, 0, sizeof(desc));
    desc.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.dimensionCount = 4;
    desc.pDimensions = dims;

    VkMLGraphNodeCreateInfoKHR node;
    VkMLTensorBindingKHR input, output;
    VkMLGraphCreateInfoKHR ci = make_single_node_graph(&node, &input, &output, &desc, &desc);
    node.operationType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR;
    node.pOperationDesc = &pool;

    VkBool32 r = vk_ml_validate_graph_create(&ci, &features, &props);
    expect("graph_rejects_invalid_pooling_node", r, VK_FALSE);
}

static void test_graph_rejects_invalid_norm_node(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();

    VkMLPrimitiveDescNormalizationKHR norm;
    memset(&norm, 0, sizeof(norm));
    norm.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR;
    norm.normType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR;
    norm.epsilon = -1.0f; /* invalid */

    static const uint32_t dims[] = {1, 1, 4, 4};
    VkTensorDescriptionKHR desc;
    memset(&desc, 0, sizeof(desc));
    desc.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.dimensionCount = 4;
    desc.pDimensions = dims;

    VkMLGraphNodeCreateInfoKHR node;
    VkMLTensorBindingKHR input, output;
    VkMLGraphCreateInfoKHR ci = make_single_node_graph(&node, &input, &output, &desc, &desc);
    node.operationType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR;
    node.pOperationDesc = &norm;

    VkBool32 r = vk_ml_validate_graph_create(&ci, &features, &props);
    expect("graph_rejects_invalid_norm_node", r, VK_FALSE);
}

/* ================================================================== */
/* Tensor Create Sharing Mode Validation                              */
/* ================================================================== */

static void test_tensor_create_concurrent_no_indices(void)
{
    static const uint32_t dims[] = {4, 4};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .dimensionCount = 2,
        .pDimensions = dims,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();
    VkBool32 r = vk_ml_validate_tensor_create(&ci, &features, &props);
    expect("tensor_create_concurrent_no_indices", r, VK_FALSE);
}

static void test_tensor_create_concurrent_valid(void)
{
    static const uint32_t dims[] = {4, 4};
    static const uint32_t queueFamilies[] = {0, 1};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .dimensionCount = 2,
        .pDimensions = dims,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = 2,
        .pQueueFamilyIndices = queueFamilies,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();
    VkBool32 r = vk_ml_validate_tensor_create(&ci, &features, &props);
    expect("tensor_create_concurrent_valid", r, VK_TRUE);
}

/* ================================================================== */
/* Boundary Tests                                                     */
/* ================================================================== */

static void test_tensor_max_dimensions(void)
{
    uint32_t dims[8];
    for (int i = 0; i < 8; i++) dims[i] = 2;

    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .dimensionCount = 8,
        .pDimensions = dims,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();
    VkBool32 r = vk_ml_validate_tensor_create(&ci, &features, &props);
    expect("tensor_max_dimensions_8d", r, VK_TRUE);
}

static void test_tensor_exceed_max_dimensions(void)
{
    uint32_t dims[9];
    for (int i = 0; i < 9; i++) dims[i] = 2;

    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .dimensionCount = 9,
        .pDimensions = dims,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };
    VkTensorCreateInfoKHR ci = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
        .pDescription = &desc,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();
    VkBool32 r = vk_ml_validate_tensor_create(&ci, &features, &props);
    expect("tensor_exceed_max_dimensions_9d", r, VK_FALSE);
}

static void test_graph_max_nodes(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = make_features();
    VkPhysicalDeviceMLPropertiesKHR props = make_props();

    static const uint32_t dims[] = {1, 1, 4, 4};
    VkTensorDescriptionKHR desc;
    memset(&desc, 0, sizeof(desc));
    desc.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.dimensionCount = 4;
    desc.pDimensions = dims;

    VkMLTensorBindingKHR binding;
    memset(&binding, 0, sizeof(binding));
    binding.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    binding.bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR;
    binding.pTensorDescription = &desc;

    uint32_t maxNodes = props.maxMLGraphNodeCount;
    VkMLGraphNodeCreateInfoKHR nodes[64];
    if (maxNodes > 64) maxNodes = 64;
    for (uint32_t i = 0; i < maxNodes; i++) {
        memset(&nodes[i], 0, sizeof(nodes[i]));
        nodes[i].sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR;
        nodes[i].operationType = VK_ML_OPERATION_TYPE_RELU_KHR;
        nodes[i].inputCount = 1;
        nodes[i].pInputs = &binding;
        nodes[i].outputCount = 1;
        nodes[i].pOutputs = &binding;
    }

    VkMLGraphCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR;
    ci.nodeCount = maxNodes;
    ci.pNodes = nodes;
    ci.externalInputCount = 1;
    ci.pExternalInputDescriptions = &desc;
    ci.externalOutputCount = 1;
    ci.pExternalOutputDescriptions = &desc;

    VkBool32 r = vk_ml_validate_graph_create(&ci, &features, &props);
    expect("graph_max_nodes_accepted", r, VK_TRUE);
}

/* ================================================================== */
/* Barrier Integration Tests                                          */
/* ================================================================== */

static void test_barrier_valid(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = (VkTensorKHR)(uintptr_t)0x1,
    };
    VkBool32 r = vk_ml_validate_tensor_memory_barrier(&barrier);
    expect("barrier_valid", r, VK_TRUE);
}

static void test_barrier_dependency_info_valid(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = (VkTensorKHR)(uintptr_t)0x1,
    };
    VkTensorDependencyInfoKHR depInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR,
        .pNext = NULL,
        .tensorMemoryBarrierCount = 1,
        .pTensorMemoryBarriers = &barrier,
    };
    VkBool32 r = vk_ml_validate_tensor_dependency_info(&depInfo);
    expect("barrier_dependency_info_valid", r, VK_TRUE);
}

static void test_barrier_dependency_info_null_barriers(void)
{
    VkTensorDependencyInfoKHR depInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR,
        .pNext = NULL,
        .tensorMemoryBarrierCount = 1,
        .pTensorMemoryBarriers = NULL,
    };
    VkBool32 r = vk_ml_validate_tensor_dependency_info(&depInfo);
    expect("barrier_dependency_info_null_barriers", r, VK_FALSE);
}

static void test_barrier_dependency_info_invalid_barrier(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = VK_NULL_HANDLE, /* invalid */
    };
    VkTensorDependencyInfoKHR depInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_KHR,
        .pNext = NULL,
        .tensorMemoryBarrierCount = 1,
        .pTensorMemoryBarriers = &barrier,
    };
    VkBool32 r = vk_ml_validate_tensor_dependency_info(&depInfo);
    expect("barrier_dependency_info_invalid_barrier", r, VK_FALSE);
}

/* ================================================================== */
/* ICD Graph Create with Invalid Descriptor Test                      */
/* ================================================================== */

static void test_icd_graph_create_rejects_invalid_desc(void)
{
    VkMLPrimitiveDescConvolutionKHR conv;
    memset(&conv, 0, sizeof(conv));
    conv.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR;
    conv.strideX = 0;
    conv.strideY = 0;
    conv.dilationX = 0;
    conv.dilationY = 0;
    conv.groupCount = 0;

    static const uint32_t dims[] = {1, 1, 4, 4};
    VkTensorDescriptionKHR desc;
    memset(&desc, 0, sizeof(desc));
    desc.sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.dimensionCount = 4;
    desc.pDimensions = dims;

    VkMLTensorBindingKHR binding;
    memset(&binding, 0, sizeof(binding));
    binding.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    binding.bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR;
    binding.pTensorDescription = &desc;

    VkMLGraphNodeCreateInfoKHR node;
    memset(&node, 0, sizeof(node));
    node.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR;
    node.operationType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR;
    node.pOperationDesc = &conv;
    node.inputCount = 1;
    node.pInputs = &binding;
    node.outputCount = 1;
    node.pOutputs = &binding;

    VkMLGraphCreateInfoKHR ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR;
    ci.nodeCount = 1;
    ci.pNodes = &node;
    ci.externalInputCount = 1;
    ci.pExternalInputDescriptions = &desc;
    ci.externalOutputCount = 1;
    ci.pExternalOutputDescriptions = &desc;

    VkMLGraphKHR graph = VK_NULL_HANDLE;
    VkResult r = vkCreateMLGraphKHR(VK_NULL_HANDLE, &ci, NULL, &graph);
    expect_vk("icd_graph_create_rejects_invalid_desc", r, VK_ERROR_UNKNOWN);
}

/* ================================================================== */
/* Main                                                               */
/* ================================================================== */

int main(void)
{
    passed = 0;
    failed = 0;

    /* Tensor view validation */
    test_tensor_view_valid();
    test_tensor_view_unbound_memory();
    test_tensor_view_format_mismatch();
    test_tensor_view_dim_count_mismatch();
    test_tensor_view_range_overflow();
    test_tensor_view_null_create_info();

    /* Tensor copy validation */
    test_tensor_copy_valid();
    test_tensor_copy_same_tensor();
    test_tensor_copy_zero_regions();
    test_tensor_copy_null_regions_ptr();
    test_tensor_copy_null_src_offsets();
    test_tensor_copy_null_info();

    /* Session validation */
    test_session_valid_explicit_scratch();
    test_session_null_graph();
    test_session_insufficient_scratch();
    test_session_auto_alloc_without_feature();
    test_session_auto_alloc_with_feature();
    test_session_null_info();

    /* Dispatch validation */
    test_dispatch_valid();
    test_dispatch_null_session();
    test_dispatch_zero_inputs();
    test_dispatch_zero_outputs();
    test_dispatch_null_input_ptr();
    test_dispatch_null_output_ptr();
    test_dispatch_weights_nonzero_null_ptr();
    test_dispatch_null_info();

    /* Graph per-node descriptor validation */
    test_graph_rejects_invalid_conv_node();
    test_graph_accepts_valid_conv_node();
    test_graph_rejects_invalid_pooling_node();
    test_graph_rejects_invalid_norm_node();

    /* Tensor create sharing mode */
    test_tensor_create_concurrent_no_indices();
    test_tensor_create_concurrent_valid();

    /* Boundary tests */
    test_tensor_max_dimensions();
    test_tensor_exceed_max_dimensions();
    test_graph_max_nodes();

    /* Barrier integration tests */
    test_barrier_valid();
    test_barrier_dependency_info_valid();
    test_barrier_dependency_info_null_barriers();
    test_barrier_dependency_info_invalid_barrier();

    /* ICD-level graph create with invalid descriptor */
    test_icd_graph_create_rejects_invalid_desc();

    printf("\nValidation coverage tests: %d passed, %d failed\n",
           passed, failed);
    return failed == 0 ? 0 : 1;
}
