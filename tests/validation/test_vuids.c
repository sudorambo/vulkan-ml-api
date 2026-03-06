/**
 * @file test_vuids.c
 * @brief Validation layer VUID tests for VK_KHR_ml_primitives.
 */

#include "../../layers/validation/vk_ml_validation.h"
#include "../../src/internal.h"
#include <math.h>
#include <stdio.h>

extern void vk_ml_populate_features(VkPhysicalDeviceMLFeaturesKHR *features);
extern void vk_ml_populate_properties(VkPhysicalDeviceMLPropertiesKHR *props);

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

static void test_graph_valid(void);
static void test_graph_zero_nodes(void);
static void test_graph_exceed_max_nodes(void);
static void test_graph_cyclic_vuid(void);
static void test_conv_zero_stride_vuid(void);
static void test_gemm_inf_alpha_vuid(void);
static void test_pool_zero_window_vuid(void);
static void test_norm_zero_epsilon_vuid(void);
static void test_elem_invalid_op_vuid(void);
static void test_tensor_double_bind(void);
static void test_tensor_bind_misaligned(void);
static void test_tensor_bind_null_memory(void);
static void test_barrier_null_tensor_vuid(void);
static void test_barrier_asymmetric_qf_vuid(void);

extern VkBool32 vk_ml_validate_tensor_memory_barrier(const VkTensorMemoryBarrierKHR *barrier);

int main(void)
{
    passed = 0;
    failed = 0;

    /* Valid tensor create */
    {
        static const uint32_t dims[] = {4, 4};
        VkTensorDescriptionKHR desc = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
            .pNext = NULL,
            .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
            .format = VK_FORMAT_R32_SFLOAT,
            .dimensionCount = 2,
            .pDimensions = dims,
            .pStrides = NULL,
            .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
        };
        VkTensorCreateInfoKHR createInfo = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
            .pNext = NULL,
            .flags = 0,
            .pDescription = &desc,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = NULL,
        };
        VkPhysicalDeviceMLFeaturesKHR features = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
            .pNext = NULL,
            .tensorObjects = VK_TRUE,
        };
        VkPhysicalDeviceMLPropertiesKHR props = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
            .pNext = NULL,
            .maxTensorDimensions = 8,
            .maxTensorElements = 1ULL << 32,
            .maxTensorDimensionSize = 65536,
        };
        VkBool32 r = vk_ml_validate_tensor_create(&createInfo, &features, &props);
        expect("vk_ml_validate_tensor_create valid", r, VK_TRUE);
    }

    /* dimensionCount = 0 */
    {
        VkTensorDescriptionKHR desc = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
            .pNext = NULL,
            .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
            .format = VK_FORMAT_R32_SFLOAT,
            .dimensionCount = 0,
            .pDimensions = NULL,
            .pStrides = NULL,
            .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
        };
        VkTensorCreateInfoKHR createInfo = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
            .pNext = NULL,
            .flags = 0,
            .pDescription = &desc,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = NULL,
        };
        VkPhysicalDeviceMLFeaturesKHR features = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
            .pNext = NULL,
            .tensorObjects = VK_TRUE,
        };
        VkPhysicalDeviceMLPropertiesKHR props = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
            .pNext = NULL,
            .maxTensorDimensions = 8,
            .maxTensorElements = 1ULL << 32,
            .maxTensorDimensionSize = 65536,
        };
        VkBool32 r = vk_ml_validate_tensor_create(&createInfo, &features, &props);
        expect("vk_ml_validate_tensor_create dimensionCount=0", r, VK_FALSE);
    }

    /* dimensionCount = 9 (> max 8) */
    {
        static const uint32_t dims[] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
        VkTensorDescriptionKHR desc = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
            .pNext = NULL,
            .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
            .format = VK_FORMAT_R32_SFLOAT,
            .dimensionCount = 9,
            .pDimensions = dims,
            .pStrides = NULL,
            .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
        };
        VkTensorCreateInfoKHR createInfo = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_KHR,
            .pNext = NULL,
            .flags = 0,
            .pDescription = &desc,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = NULL,
        };
        VkPhysicalDeviceMLFeaturesKHR features = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
            .pNext = NULL,
            .tensorObjects = VK_TRUE,
        };
        VkPhysicalDeviceMLPropertiesKHR props = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
            .pNext = NULL,
            .maxTensorDimensions = 8,
            .maxTensorElements = 1ULL << 32,
            .maxTensorDimensionSize = 65536,
        };
        VkBool32 r = vk_ml_validate_tensor_create(&createInfo, &features, &props);
        expect("vk_ml_validate_tensor_create dimensionCount=9", r, VK_FALSE);
    }

    /* Convolution strideX = 0 */
    {
        VkMLPrimitiveDescConvolutionKHR conv = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
            .pNext = NULL,
            .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
            .kernelWidth = 3,
            .kernelHeight = 3,
            .strideX = 0,
            .strideY = 1,
            .dilationX = 1,
            .dilationY = 1,
            .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
            .paddingTop = 0,
            .paddingBottom = 0,
            .paddingLeft = 0,
            .paddingRight = 0,
            .groupCount = 1,
            .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
            .activationParam0 = 0.0f,
            .activationParam1 = 0.0f,
        };
        VkPhysicalDeviceMLFeaturesKHR features = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
            .pNext = NULL,
            .mlPrimitives = VK_TRUE,
        };
        VkBool32 r = vk_ml_validate_convolution_desc(&conv, &features);
        expect("vk_ml_validate_convolution_desc strideX=0", r, VK_FALSE);
    }

    /* GEMM alpha = INFINITY */
    {
        VkMLPrimitiveDescGemmKHR gemm = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR,
            .pNext = NULL,
            .transposeA = VK_FALSE,
            .transposeB = VK_FALSE,
            .alpha = (float)INFINITY,
            .beta = 0.0f,
            .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
            .activationParam0 = 0.0f,
            .activationParam1 = 0.0f,
        };
        VkPhysicalDeviceMLFeaturesKHR features = {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
            .pNext = NULL,
            .mlPrimitives = VK_TRUE,
        };
        VkBool32 r = vk_ml_validate_gemm_desc(&gemm, &features);
        expect("vk_ml_validate_gemm_desc alpha=INFINITY", r, VK_FALSE);
    }

    /* Graph validation tests */
    test_graph_valid();
    test_graph_zero_nodes();
    test_graph_exceed_max_nodes();
    test_graph_cyclic_vuid();
    test_conv_zero_stride_vuid();
    test_gemm_inf_alpha_vuid();
    test_pool_zero_window_vuid();
    test_norm_zero_epsilon_vuid();
    test_elem_invalid_op_vuid();
    test_tensor_double_bind();
    test_tensor_bind_misaligned();
    test_tensor_bind_null_memory();
    test_barrier_null_tensor_vuid();
    test_barrier_asymmetric_qf_vuid();

    (void)printf("\nTotal: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* Graph VUID test functions                                          */
/* ------------------------------------------------------------------ */

static void test_graph_valid(void)
{
    static const uint32_t dims[] = {4, 4};
    VkTensorDescriptionKHR extInDesc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR extOutDesc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };
    VkMLTensorBindingKHR inputBinding = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &extInDesc,
    };
    VkMLTensorBindingKHR outputBinding = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &extOutDesc,
    };
    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = NULL,
        .inputCount = 1,
        .pInputs = &inputBinding,
        .outputCount = 1,
        .pOutputs = &outputBinding,
        .pNodeName = NULL,
    };
    VkMLGraphCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &extInDesc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &extOutDesc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlGraph = VK_TRUE,
    };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
        .maxMLGraphNodeCount = VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT,
    };
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);
    VkBool32 r = vk_ml_validate_graph_create(&createInfo, &features, &props);
    expect("test_graph_valid", r, VK_TRUE);
}

static void test_graph_zero_nodes(void)
{
    VkMLGraphCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 0,
        .pNodes = NULL,
        .externalInputCount = 1,
        .pExternalInputDescriptions = NULL,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = NULL,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlGraph = VK_TRUE,
    };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
        .maxMLGraphNodeCount = VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT,
    };
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);
    VkBool32 r = vk_ml_validate_graph_create(&createInfo, &features, &props);
    expect("test_graph_zero_nodes", r, VK_FALSE);
}

static void test_graph_exceed_max_nodes(void)
{
    VkMLGraphCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 300,
        .pNodes = NULL,
        .externalInputCount = 1,
        .pExternalInputDescriptions = NULL,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = NULL,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlGraph = VK_TRUE,
    };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
        .maxMLGraphNodeCount = VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT,
    };
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);
    VkBool32 r = vk_ml_validate_graph_create(&createInfo, &features, &props);
    expect("test_graph_exceed_max_nodes", r, VK_FALSE);
}

static void test_graph_cyclic_vuid(void)
{
    static const uint32_t dims[] = {4, 4};
    VkTensorDescriptionKHR extInDesc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR extOutDesc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 2,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    /* Node 0 input from node 1 (INTERNAL), node 1 input from node 0 (INTERNAL) = cycle */
    VkMLTensorBindingKHR node0_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 1,
        .tensorIndex = 0,
        .pTensorDescription = &extInDesc,
    };
    VkMLTensorBindingKHR node0_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &extOutDesc,
    };
    VkMLTensorBindingKHR node1_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &extInDesc,
    };
    VkMLTensorBindingKHR node1_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 1,
        .tensorIndex = 0,
        .pTensorDescription = &extOutDesc,
    };

    VkMLGraphNodeCreateInfoKHR nodes[2] = {
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1,
            .pInputs = &node0_input,
            .outputCount = 1,
            .pOutputs = &node0_output,
            .pNodeName = NULL,
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1,
            .pInputs = &node1_input,
            .outputCount = 1,
            .pOutputs = &node1_output,
            .pNodeName = NULL,
        },
    };

    VkMLGraphCreateInfoKHR createInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 2,
        .pNodes = nodes,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &extInDesc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &extOutDesc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlGraph = VK_TRUE,
    };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
        .maxMLGraphNodeCount = VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT,
    };
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);
    VkBool32 r = vk_ml_validate_graph_create(&createInfo, &features, &props);
    expect("test_graph_cyclic_vuid", r, VK_FALSE);
}

static void test_conv_zero_stride_vuid(void)
{
    VkMLPrimitiveDescConvolutionKHR conv = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
        .pNext = NULL,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .kernelWidth = 3,
        .kernelHeight = 3,
        .strideX = 0,
        .strideY = 1,
        .dilationX = 1,
        .dilationY = 1,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
        .groupCount = 1,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlPrimitives = VK_TRUE,
    };
    VkBool32 r = vk_ml_validate_convolution_desc(&conv, &features);
    expect("test_conv_zero_stride_vuid", r, VK_FALSE);
}

static void test_gemm_inf_alpha_vuid(void)
{
    VkMLPrimitiveDescGemmKHR gemm = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR,
        .pNext = NULL,
        .transposeA = VK_FALSE,
        .transposeB = VK_FALSE,
        .alpha = (float)INFINITY,
        .beta = 0.0f,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlPrimitives = VK_TRUE,
    };
    VkBool32 r = vk_ml_validate_gemm_desc(&gemm, &features);
    expect("test_gemm_inf_alpha_vuid", r, VK_FALSE);
}

static void test_pool_zero_window_vuid(void)
{
    VkMLPrimitiveDescPoolingKHR pool = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 0,
        .windowHeight = 2,
        .strideX = 1,
        .strideY = 1,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
    };
    VkBool32 r = vk_ml_validate_pooling_desc(&pool);
    expect("test_pool_zero_window_vuid", r, VK_FALSE);
}

static void test_norm_zero_epsilon_vuid(void)
{
    VkMLPrimitiveDescNormalizationKHR norm = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR,
        .pNext = NULL,
        .normType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR,
        .epsilon = 0.0f,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlPrimitives = VK_TRUE,
    };
    VkBool32 r = vk_ml_validate_normalization_desc(&norm, &features);
    expect("test_norm_zero_epsilon_vuid", r, VK_FALSE);
}

static void test_elem_invalid_op_vuid(void)
{
    VkMLPrimitiveDescElementwiseKHR elem = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR,
        .pNext = NULL,
        .opType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
        .mlPrimitives = VK_TRUE,
    };
    VkBool32 r = vk_ml_validate_elementwise_desc(&elem, &features);
    expect("test_elem_invalid_op_vuid", r, VK_FALSE);
}

/* ------------------------------------------------------------------ */
/* Barrier VUID test functions                                         */
/* ------------------------------------------------------------------ */

static void test_barrier_null_tensor_vuid(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .tensor = VK_NULL_HANDLE,
    };
    VkBool32 r = vk_ml_validate_tensor_memory_barrier(&barrier);
    expect("test_barrier_null_tensor_vuid", r, VK_FALSE);
}

static void test_barrier_asymmetric_qf_vuid(void)
{
    VkTensorMemoryBarrierKHR barrier = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_KHR,
        .pNext = NULL,
        .srcAccessMask = VK_ACCESS_2_ML_GRAPH_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = 0,
        .tensor = (VkTensorKHR)(uintptr_t)1,
    };
    VkBool32 r = vk_ml_validate_tensor_memory_barrier(&barrier);
    expect("test_barrier_asymmetric_qf_vuid", r, VK_FALSE);
}

static void test_tensor_double_bind(void)
{
    VkBindTensorMemoryInfoKHR bindInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
        .pNext = NULL,
        .tensor = VK_NULL_HANDLE,
        .memory = (VkDeviceMemory)(uintptr_t)0xDEAD,
        .memoryOffset = 0,
    };
    VkTensorKHR_T tensor = { .memoryBound = VK_TRUE };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
        .maxTensorDimensions = 8,
        .maxTensorElements = 1ULL << 32,
        .maxTensorDimensionSize = 65536,
        .minTensorMemoryAlignment = 64,
    };
    VkBool32 r = vk_ml_validate_tensor_bind(&bindInfo, &tensor, &props);
    expect("test_tensor_double_bind", r, VK_FALSE);
}

static void test_tensor_bind_misaligned(void)
{
    VkBindTensorMemoryInfoKHR bindInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
        .pNext = NULL,
        .tensor = VK_NULL_HANDLE,
        .memory = (VkDeviceMemory)(uintptr_t)0xDEAD,
        .memoryOffset = 3,
    };
    VkTensorKHR_T tensor = { .memoryBound = VK_FALSE };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
        .maxTensorDimensions = 8,
        .maxTensorElements = 1ULL << 32,
        .maxTensorDimensionSize = 65536,
        .minTensorMemoryAlignment = 64,
    };
    VkBool32 r = vk_ml_validate_tensor_bind(&bindInfo, &tensor, &props);
    expect("test_tensor_bind_misaligned", r, VK_FALSE);
}

static void test_tensor_bind_null_memory(void)
{
    VkBindTensorMemoryInfoKHR bindInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_KHR,
        .pNext = NULL,
        .tensor = VK_NULL_HANDLE,
        .memory = VK_NULL_HANDLE,
        .memoryOffset = 0,
    };
    VkTensorKHR_T tensor = { .memoryBound = VK_FALSE };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
        .maxTensorDimensions = 8,
        .maxTensorElements = 1ULL << 32,
        .maxTensorDimensionSize = 65536,
        .minTensorMemoryAlignment = 64,
    };
    VkBool32 r = vk_ml_validate_tensor_bind(&bindInfo, &tensor, &props);
    expect("test_tensor_bind_null_memory", r, VK_FALSE);
}
