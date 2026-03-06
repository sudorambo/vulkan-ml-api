/**
 * @file test_tensor_formats.c
 * @brief CTS tests for VK_KHR_ml_primitives format queries.
 *
 * Verifies format support and feature/property population.
 * Tests feature_query.c functions directly.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/* Extern declarations for feature_query.c */
extern void vk_ml_populate_features(VkPhysicalDeviceMLFeaturesKHR *features);
extern void vk_ml_populate_properties(VkPhysicalDeviceMLPropertiesKHR *props);
extern VkBool32 vk_ml_is_tensor_format_supported(VkFormat format);
extern void vk_ml_populate_tensor_format_properties(VkFormat format,
    VkTensorFormatPropertiesKHR *props);

static int g_fail_count = 0;

#define RUN_TEST(name) do { \
    printf("Running %s...\n", #name); \
    if (name()) { printf("FAIL: %s\n", #name); g_fail_count++; } \
    else { printf("PASS: %s\n", #name); } \
} while (0)

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static int test_format_support_fp16(void)
{
    return vk_ml_is_tensor_format_supported(VK_FORMAT_R16_SFLOAT) == VK_TRUE ? 0 : 1;
}

static int test_format_support_bf16(void)
{
    return vk_ml_is_tensor_format_supported((VkFormat)VK_FORMAT_R16_BFLOAT_KHR) == VK_TRUE
        ? 0 : 1;
}

static int test_format_support_fp8(void)
{
    if (vk_ml_is_tensor_format_supported((VkFormat)VK_FORMAT_R8_E4M3_KHR) != VK_TRUE)
        return 1;
    if (vk_ml_is_tensor_format_supported((VkFormat)VK_FORMAT_R8_E5M2_KHR) != VK_TRUE)
        return 1;
    return 0;
}

static int test_format_unsupported(void)
{
    /* Use a format not in the supported list (e.g. R64_SFLOAT) */
    VkTensorFormatPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_KHR,
        .pNext = NULL,
        .tensorFeatures = 0xFFFFFFFFu,
    };
    vk_ml_populate_tensor_format_properties(VK_FORMAT_R64_SFLOAT, &props);
    return (props.tensorFeatures == 0) ? 0 : 1;
}

static int test_populate_features(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    if (features.mlPrimitives != VK_TRUE) return 1;
    if (features.mlGraph != VK_TRUE) return 1;
    if (features.tensorObjects != VK_TRUE) return 1;
    if (features.tensorShaderAccess != VK_TRUE) return 1;
    if (features.tensorImageAliasing != VK_TRUE) return 1;
    if (features.fp16Tensors != VK_TRUE) return 1;
    if (features.bf16Tensors != VK_TRUE) return 1;
    if (features.int8Tensors != VK_TRUE) return 1;
    if (features.int4Tensors != VK_TRUE) return 1;
    if (features.fp8Tensors != VK_TRUE) return 1;
    if (features.fusedActivations != VK_TRUE) return 1;
    if (features.mlGraphScratchAutoAllocation != VK_TRUE) return 1;
    return 0;
}

static int test_populate_properties(void)
{
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_properties(&props);
    return (props.maxTensorDimensions == 8) ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* pNext chain preservation tests (C2 remediation)                    */
/* ------------------------------------------------------------------ */

static int test_features_pnext_preserved(void)
{
    VkPhysicalDeviceMLPropertiesKHR chain = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = NULL,
    };
    VkPhysicalDeviceMLFeaturesKHR features = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = &chain,
    };
    vk_ml_populate_features(&features);

    if (features.sType != (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR)
        return 1;
    if (features.pNext != &chain)
        return 1;
    return 0;
}

static int test_properties_pnext_preserved(void)
{
    VkPhysicalDeviceMLFeaturesKHR chain = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
    };
    VkPhysicalDeviceMLPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR,
        .pNext = &chain,
    };
    vk_ml_populate_properties(&props);

    if (props.sType != (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_PROPERTIES_KHR)
        return 1;
    if (props.pNext != &chain)
        return 1;
    return 0;
}

static int test_format_props_pnext_preserved(void)
{
    VkPhysicalDeviceMLFeaturesKHR chain = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ML_FEATURES_KHR,
        .pNext = NULL,
    };
    VkTensorFormatPropertiesKHR props = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_KHR,
        .pNext = &chain,
    };
    vk_ml_populate_tensor_format_properties(VK_FORMAT_R16_SFLOAT, &props);

    if (props.sType != (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_KHR)
        return 1;
    if (props.pNext != &chain)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(test_format_support_fp16);
    RUN_TEST(test_format_support_bf16);
    RUN_TEST(test_format_support_fp8);
    RUN_TEST(test_format_unsupported);
    RUN_TEST(test_populate_features);
    RUN_TEST(test_populate_properties);
    RUN_TEST(test_features_pnext_preserved);
    RUN_TEST(test_properties_pnext_preserved);
    RUN_TEST(test_format_props_pnext_preserved);

    if (g_fail_count > 0) {
        printf("\n%d test(s) failed.\n", g_fail_count);
        return 1;
    }
    printf("\nAll tests passed.\n");
    return 0;
}
