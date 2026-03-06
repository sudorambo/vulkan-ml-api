/**
 * @file test_descriptor_validation.c
 * @brief Descriptor validation unit tests (US2) - primitive descriptor validation.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "../../layers/validation/vk_ml_validation.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

extern void vk_ml_populate_features(VkPhysicalDeviceMLFeaturesKHR *features);

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

static void test_valid_convolution(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescConvolutionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
        .pNext = NULL,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .kernelWidth = 3,
        .kernelHeight = 3,
        .strideX = 1,
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

    VkBool32 r = vk_ml_validate_convolution_desc(&desc, &features);
    expect("test_valid_convolution", r, VK_TRUE);
}

static void test_conv_zero_stride(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescConvolutionKHR desc = {
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

    VkBool32 r = vk_ml_validate_convolution_desc(&desc, &features);
    expect("test_conv_zero_stride", r, VK_FALSE);
}

static void test_conv_zero_dilation(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescConvolutionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_CONVOLUTION_KHR,
        .pNext = NULL,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .kernelWidth = 3,
        .kernelHeight = 3,
        .strideX = 1,
        .strideY = 1,
        .dilationX = 0,
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

    VkBool32 r = vk_ml_validate_convolution_desc(&desc, &features);
    expect("test_conv_zero_dilation", r, VK_FALSE);
}

static void test_valid_gemm(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescGemmKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR,
        .pNext = NULL,
        .transposeA = VK_FALSE,
        .transposeB = VK_FALSE,
        .alpha = 1.0f,
        .beta = 0.0f,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_gemm_desc(&desc, &features);
    expect("test_valid_gemm", r, VK_TRUE);
}

static void test_gemm_infinite_alpha(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescGemmKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR,
        .pNext = NULL,
        .transposeA = VK_FALSE,
        .transposeB = VK_FALSE,
        .alpha = INFINITY,
        .beta = 0.0f,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_gemm_desc(&desc, &features);
    expect("test_gemm_infinite_alpha", r, VK_FALSE);
}

static void test_gemm_nan_beta(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescGemmKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_GEMM_KHR,
        .pNext = NULL,
        .transposeA = VK_FALSE,
        .transposeB = VK_FALSE,
        .alpha = 1.0f,
        .beta = NAN,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_gemm_desc(&desc, &features);
    expect("test_gemm_nan_beta", r, VK_FALSE);
}

static void test_valid_pooling(void)
{
    VkMLPrimitiveDescPoolingKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 2,
        .windowHeight = 2,
        .strideX = 2,
        .strideY = 2,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
    };

    VkBool32 r = vk_ml_validate_pooling_desc(&desc);
    expect("test_valid_pooling", r, VK_TRUE);
}

static void test_pool_zero_window(void)
{
    VkMLPrimitiveDescPoolingKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 0,
        .windowHeight = 2,
        .strideX = 2,
        .strideY = 2,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
    };

    VkBool32 r = vk_ml_validate_pooling_desc(&desc);
    expect("test_pool_zero_window", r, VK_FALSE);
}

static void test_pool_zero_stride(void)
{
    VkMLPrimitiveDescPoolingKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_MAX_POOL_2D_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 2,
        .windowHeight = 2,
        .strideX = 0,
        .strideY = 2,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
    };

    VkBool32 r = vk_ml_validate_pooling_desc(&desc);
    expect("test_pool_zero_stride", r, VK_FALSE);
}

static void test_pool_invalid_type(void)
{
    VkMLPrimitiveDescPoolingKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_POOLING_KHR,
        .pNext = NULL,
        .poolType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .windowWidth = 2,
        .windowHeight = 2,
        .strideX = 2,
        .strideY = 2,
        .paddingMode = VK_ML_PADDING_MODE_VALID_KHR,
        .paddingTop = 0,
        .paddingBottom = 0,
        .paddingLeft = 0,
        .paddingRight = 0,
    };

    VkBool32 r = vk_ml_validate_pooling_desc(&desc);
    expect("test_pool_invalid_type", r, VK_FALSE);
}

static void test_valid_normalization(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescNormalizationKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR,
        .pNext = NULL,
        .normType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR,
        .epsilon = 1e-5f,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_normalization_desc(&desc, &features);
    expect("test_valid_normalization", r, VK_TRUE);
}

static void test_norm_zero_epsilon(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescNormalizationKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR,
        .pNext = NULL,
        .normType = VK_ML_OPERATION_TYPE_BATCH_NORMALIZATION_KHR,
        .epsilon = 0.0f,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_normalization_desc(&desc, &features);
    expect("test_norm_zero_epsilon", r, VK_FALSE);
}

static void test_norm_invalid_type(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescNormalizationKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_NORMALIZATION_KHR,
        .pNext = NULL,
        .normType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .epsilon = 1e-5f,
        .inputLayout = VK_ML_TENSOR_LAYOUT_NCHW_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_normalization_desc(&desc, &features);
    expect("test_norm_invalid_type", r, VK_FALSE);
}

static void test_valid_elementwise(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescElementwiseKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR,
        .pNext = NULL,
        .opType = VK_ML_OPERATION_TYPE_ELEMENTWISE_ADD_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_elementwise_desc(&desc, &features);
    expect("test_valid_elementwise", r, VK_TRUE);
}

static void test_elem_invalid_op(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkMLPrimitiveDescElementwiseKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_PRIMITIVE_DESC_ELEMENTWISE_KHR,
        .pNext = NULL,
        .opType = VK_ML_OPERATION_TYPE_CONVOLUTION_2D_KHR,
        .fusedActivation = VK_ML_ACTIVATION_FUNCTION_NONE_KHR,
        .activationParam0 = 0.0f,
        .activationParam1 = 0.0f,
    };

    VkBool32 r = vk_ml_validate_elementwise_desc(&desc, &features);
    expect("test_elem_invalid_op", r, VK_FALSE);
}

static void test_dimension_product_overflow(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkPhysicalDeviceMLPropertiesKHR props = {0};
    props.maxTensorDimensions = 8;
    props.maxTensorDimensionSize = 65536;
    props.maxTensorElements = (1ULL << 32);

    uint32_t dims[] = {65536, 65536, 65536, 65536};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 4,
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

    VkBool32 r = vk_ml_validate_tensor_create(&createInfo, &features, &props);
    expect("test_dimension_product_overflow", r, VK_FALSE);
}

int main(void)
{
    passed = 0;
    failed = 0;

    test_valid_convolution();
    test_conv_zero_stride();
    test_conv_zero_dilation();
    test_valid_gemm();
    test_gemm_infinite_alpha();
    test_gemm_nan_beta();
    test_valid_pooling();
    test_pool_zero_window();
    test_pool_zero_stride();
    test_pool_invalid_type();
    test_valid_normalization();
    test_norm_zero_epsilon();
    test_norm_invalid_type();
    test_valid_elementwise();
    test_elem_invalid_op();
    test_dimension_product_overflow();

    (void)printf("\nTotal: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
