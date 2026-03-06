/**
 * @file test_spirv_tensor.c
 * @brief SPIR-V Tensor Shader Access CTS tests (US5).
 *
 * Verifies descriptor type and write descriptor set structures are correctly
 * constructable. Cannot compile SPIR-V or run shaders in the reference impl.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include <stdio.h>
#include <string.h>

/* Extern declarations from feature_query.c */
extern void vk_ml_populate_features(VkPhysicalDeviceMLFeaturesKHR *features);
extern void vk_ml_populate_tensor_format_properties(VkFormat format,
    VkTensorFormatPropertiesKHR *props);

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

/* ------------------------------------------------------------------ */
/* test_descriptor_type_constant                                       */
/* ------------------------------------------------------------------ */

static int test_descriptor_type_constant(void)
{
    int val = VK_DESCRIPTOR_TYPE_TENSOR_KHR;
    if (val != 1000559000)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_write_descriptor_set_tensor                                    */
/* ------------------------------------------------------------------ */

static int test_write_descriptor_set_tensor(void)
{
    uint32_t dims[] = {1, 2, 4, 4};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
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
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);
    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;

    VkTensorViewCreateInfoKHR viewInfo = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .tensor = tensor,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 0,
        .pDimensionOffsets = NULL,
        .pDimensionSizes = NULL,
    };
    VkTensorViewKHR view = VK_NULL_HANDLE;
    r = vkCreateTensorViewKHR(VK_NULL_HANDLE, &viewInfo, NULL, &view);
    if (r != VK_SUCCESS || view == VK_NULL_HANDLE) {
        vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
        return 1;
    }

    VkWriteDescriptorSetTensorKHR writeTensor = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_KHR,
        .pNext = NULL,
        .tensorCount = 1,
        .pTensorViews = &view,
    };

    int ok = (writeTensor.sType == (VkStructureType)1000559008 &&
              writeTensor.tensorCount == 1 &&
              writeTensor.pTensorViews != NULL &&
              writeTensor.pTensorViews[0] == view);

    vkDestroyTensorViewKHR(VK_NULL_HANDLE, view, NULL);
    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* test_format_feature_shader_bit                                      */
/* ------------------------------------------------------------------ */

static int test_format_feature_shader_bit(void)
{
    VkTensorFormatPropertiesKHR props = {0};
    vk_ml_populate_tensor_format_properties(VK_FORMAT_R32_SFLOAT, &props);

    if ((props.tensorFeatures & (VkFormatFeatureFlags2)VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_KHR) == 0)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_format_feature_aliasing_bit                                    */
/* ------------------------------------------------------------------ */

static int test_format_feature_aliasing_bit(void)
{
    static const VkFormat supported_formats[] = {
        VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R8_SINT,
    };
    const size_t n = sizeof(supported_formats) / sizeof(supported_formats[0]);

    for (size_t i = 0; i < n; i++) {
        VkTensorFormatPropertiesKHR props = {0};
        vk_ml_populate_tensor_format_properties(supported_formats[i], &props);

        if ((props.tensorFeatures & (VkFormatFeatureFlags2)VK_FORMAT_FEATURE_2_TENSOR_IMAGE_ALIASING_BIT_KHR) == 0)
            return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_tensor_shader_access_feature                                   */
/* ------------------------------------------------------------------ */

static int test_tensor_shader_access_feature(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    if (features.tensorShaderAccess != VK_TRUE)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* test_tensor_usage_shader_bit                                        */
/* ------------------------------------------------------------------ */

static int test_tensor_usage_shader_bit(void)
{
    uint32_t dims[] = {1, 2, 4, 4};
    VkTensorDescriptionKHR desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
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
    VkTensorKHR tensor = VK_NULL_HANDLE;
    VkResult r = vkCreateTensorKHR(VK_NULL_HANDLE, &createInfo, NULL, &tensor);

    if (r != VK_SUCCESS || tensor == VK_NULL_HANDLE)
        return 1;

    vkDestroyTensorKHR(VK_NULL_HANDLE, tensor, NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_descriptor_type_constant);
    RUN_TEST(test_write_descriptor_set_tensor);
    RUN_TEST(test_format_feature_shader_bit);
    RUN_TEST(test_format_feature_aliasing_bit);
    RUN_TEST(test_tensor_shader_access_feature);
    RUN_TEST(test_tensor_usage_shader_bit);

    if (g_fail_count > 0) {
        printf("\n%d test(s) failed.\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}
