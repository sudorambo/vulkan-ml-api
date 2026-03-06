/**
 * @file test_helpers.h
 * @brief Shared test helper functions for VK_KHR_ml_primitives CTS tests.
 */

#ifndef TEST_HELPERS_H_
#define TEST_HELPERS_H_

#include <vulkan/vulkan_ml_primitives.h>

static inline void make_tensor_desc(VkTensorDescriptionKHR *desc,
                                    uint32_t *dims,
                                    uint32_t dim_count,
                                    VkFormat format,
                                    VkTensorUsageFlagsKHR usage)
{
    desc->sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR;
    desc->pNext = NULL;
    desc->tiling = VK_TENSOR_TILING_OPTIMAL_KHR;
    desc->format = format;
    desc->dimensionCount = dim_count;
    desc->pDimensions = dims;
    desc->pStrides = NULL;
    desc->usage = usage;
}

static inline void make_tensor_binding_external_input(VkMLTensorBindingKHR *b,
                                                      uint32_t tensor_index,
                                                      const VkTensorDescriptionKHR *desc)
{
    b->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    b->pNext = NULL;
    b->bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR;
    b->nodeIndex = 0;
    b->tensorIndex = tensor_index;
    b->pTensorDescription = desc;
}

static inline void make_tensor_binding_external_output(VkMLTensorBindingKHR *b,
                                                       uint32_t tensor_index,
                                                       const VkTensorDescriptionKHR *desc)
{
    b->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    b->pNext = NULL;
    b->bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR;
    b->nodeIndex = 0;
    b->tensorIndex = tensor_index;
    b->pTensorDescription = desc;
}

static inline void make_tensor_binding_external_weight(VkMLTensorBindingKHR *b,
                                                       uint32_t tensor_index,
                                                       const VkTensorDescriptionKHR *desc)
{
    b->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    b->pNext = NULL;
    b->bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_WEIGHT_KHR;
    b->nodeIndex = 0;
    b->tensorIndex = tensor_index;
    b->pTensorDescription = desc;
}

static inline void make_tensor_binding_internal(VkMLTensorBindingKHR *b,
                                                uint32_t node_index,
                                                uint32_t tensor_index,
                                                const VkTensorDescriptionKHR *desc)
{
    b->sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR;
    b->pNext = NULL;
    b->bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR;
    b->nodeIndex = node_index;
    b->tensorIndex = tensor_index;
    b->pTensorDescription = desc;
}

#endif /* TEST_HELPERS_H_ */
