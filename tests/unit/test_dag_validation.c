/**
 * @file test_dag_validation.c
 * @brief DAG validation unit tests (US2) - graph validation layer.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "internal.h"
#include "vk_ml_validation.h"
#include <stdio.h>

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

static void test_valid_single_node_graph(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t input_dims[] = {1, 4, 4, 4};
    static const uint32_t output_dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = input_dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = output_dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = NULL,
        .inputCount = 0,
        .pInputs = NULL,
        .outputCount = 0,
        .pOutputs = NULL,
        .pNodeName = NULL,
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_valid_single_node_graph", r, VK_TRUE);
}

static void test_zero_node_count(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 0,
        .pNodes = NULL,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_zero_node_count", r, VK_FALSE);
}

static void test_exceed_max_nodes(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = NULL,
        .inputCount = 0,
        .pInputs = NULL,
        .outputCount = 0,
        .pOutputs = NULL,
        .pNodeName = NULL,
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = VK_ML_REF_MAX_ML_GRAPH_NODE_COUNT + 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_exceed_max_nodes", r, VK_FALSE);
}

static void test_zero_external_inputs(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = NULL,
        .inputCount = 0,
        .pInputs = NULL,
        .outputCount = 0,
        .pOutputs = NULL,
        .pNodeName = NULL,
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 0,
        .pExternalInputDescriptions = NULL,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_zero_external_inputs", r, VK_FALSE);
}

static void test_cyclic_graph(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    /* Node 0 input from node 1 (INTERNAL), node 1 input from node 0 (INTERNAL) = A→B→A cycle */
    VkMLTensorBindingKHR node0_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 1,
        .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR node0_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };
    VkMLTensorBindingKHR node1_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR node1_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 1,
        .tensorIndex = 0,
        .pTensorDescription = &output_desc,
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

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 2,
        .pNodes = nodes,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_cyclic_graph", r, VK_FALSE);
}

static void test_self_reference(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    /* Node 0 input from node 0 (INTERNAL) = self-reference */
    VkMLTensorBindingKHR node_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR node_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = NULL,
        .inputCount = 1,
        .pInputs = &node_input,
        .outputCount = 1,
        .pOutputs = &node_output,
        .pNodeName = NULL,
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_self_reference", r, VK_FALSE);
}

static void test_invalid_node_index(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    /* Node 1 input has nodeIndex = 5 (beyond nodeCount = 2) */
    VkMLTensorBindingKHR node0_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR node0_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 0,
        .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };
    VkMLTensorBindingKHR node1_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 5,
        .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR node1_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 1,
        .tensorIndex = 0,
        .pTensorDescription = &output_desc,
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

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 2,
        .pNodes = nodes,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_invalid_node_index", r, VK_FALSE);
}

static void test_zero_external_outputs(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = NULL,
        .inputCount = 0,
        .pInputs = NULL,
        .outputCount = 0,
        .pOutputs = NULL,
        .pNodeName = NULL,
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 0,
        .pExternalOutputDescriptions = NULL,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_zero_external_outputs", r, VK_FALSE);
}

/* ------------------------------------------------------------------ */
/* T037: Diamond-shaped DAG (valid, no cycle)                          */
/* ------------------------------------------------------------------ */

static void test_diamond_dag_valid(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    /* Diamond: A(0)→B(1), A(0)→C(2), B(1)→D(3), C(2)→D(3) */
    VkMLTensorBindingKHR a_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_INPUT_KHR,
        .nodeIndex = 0, .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR a_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 0, .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };

    VkMLTensorBindingKHR b_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 0, .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR b_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 1, .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };

    VkMLTensorBindingKHR c_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 0, .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR c_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 2, .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };

    VkMLTensorBindingKHR d_inputs[2] = {
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
            .pNext = NULL,
            .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
            .nodeIndex = 1, .tensorIndex = 0,
            .pTensorDescription = &input_desc,
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
            .pNext = NULL,
            .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
            .nodeIndex = 2, .tensorIndex = 0,
            .pTensorDescription = &input_desc,
        },
    };
    VkMLTensorBindingKHR d_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 3, .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };

    VkMLGraphNodeCreateInfoKHR nodes[4] = {
        { /* A */
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &a_input,
            .outputCount = 1, .pOutputs = &a_output,
            .pNodeName = NULL,
        },
        { /* B */
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &b_input,
            .outputCount = 1, .pOutputs = &b_output,
            .pNodeName = NULL,
        },
        { /* C */
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &c_input,
            .outputCount = 1, .pOutputs = &c_output,
            .pNodeName = NULL,
        },
        { /* D */
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 2, .pInputs = d_inputs,
            .outputCount = 1, .pOutputs = &d_output,
            .pNodeName = NULL,
        },
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 4,
        .pNodes = nodes,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_diamond_dag_valid", r, VK_TRUE);
}

/* ------------------------------------------------------------------ */
/* T037: Three-node cycle (A→B→C→A)                                    */
/* ------------------------------------------------------------------ */

static void test_three_node_cycle(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    /* A(0) reads from C(2), B(1) reads from A(0), C(2) reads from B(1) → cycle */
    VkMLTensorBindingKHR a_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 2, .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR a_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 0, .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };
    VkMLTensorBindingKHR b_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 0, .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR b_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 1, .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };
    VkMLTensorBindingKHR c_input = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_INTERNAL_KHR,
        .nodeIndex = 1, .tensorIndex = 0,
        .pTensorDescription = &input_desc,
    };
    VkMLTensorBindingKHR c_output = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_TENSOR_BINDING_KHR,
        .pNext = NULL,
        .bindingType = VK_ML_TENSOR_BINDING_TYPE_EXTERNAL_OUTPUT_KHR,
        .nodeIndex = 2, .tensorIndex = 0,
        .pTensorDescription = &output_desc,
    };

    VkMLGraphNodeCreateInfoKHR nodes[3] = {
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &a_input,
            .outputCount = 1, .pOutputs = &a_output,
            .pNodeName = NULL,
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &b_input,
            .outputCount = 1, .pOutputs = &b_output,
            .pNodeName = NULL,
        },
        {
            .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
            .pNext = NULL,
            .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
            .pOperationDesc = NULL,
            .inputCount = 1, .pInputs = &c_input,
            .outputCount = 1, .pOutputs = &c_output,
            .pNodeName = NULL,
        },
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 3,
        .pNodes = nodes,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_three_node_cycle", r, VK_FALSE);
}

/* ------------------------------------------------------------------ */
/* T042: Unknown operation sType rejection                             */
/* ------------------------------------------------------------------ */

static void test_unknown_operation_stype(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_features(&features);
    vk_ml_populate_properties(&props);

    static const uint32_t dims[] = {1, 4, 4, 4};
    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_INPUT_BIT_KHR,
    };
    VkTensorDescriptionKHR output_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R32_SFLOAT,
        .dimensionCount = 4,
        .pDimensions = dims,
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_ML_GRAPH_OUTPUT_BIT_KHR,
    };

    /* Fake descriptor with sType = 0x7FFFFFFF (unknown) */
    VkStructureType fake_stype = (VkStructureType)0x7FFFFFFF;
    struct { VkStructureType sType; const void *pNext; } fake_desc = {
        .sType = fake_stype,
        .pNext = NULL,
    };

    VkMLGraphNodeCreateInfoKHR node = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_NODE_CREATE_INFO_KHR,
        .pNext = NULL,
        .operationType = VK_ML_OPERATION_TYPE_RELU_KHR,
        .pOperationDesc = &fake_desc,
        .inputCount = 0,
        .pInputs = NULL,
        .outputCount = 0,
        .pOutputs = NULL,
        .pNodeName = NULL,
    };

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = &node,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_unknown_operation_stype", r, VK_FALSE);
}

static void test_null_pnodes_with_nodecount(void)
{
    VkPhysicalDeviceMLFeaturesKHR features = {0};
    vk_ml_populate_features(&features);

    VkPhysicalDeviceMLPropertiesKHR props = {0};
    vk_ml_populate_properties(&props);

    VkTensorDescriptionKHR input_desc = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_KHR,
        .pNext = NULL,
        .tiling = VK_TENSOR_TILING_OPTIMAL_KHR,
        .format = VK_FORMAT_R16_SFLOAT,
        .dimensionCount = 1,
        .pDimensions = (uint32_t[]){4},
        .pStrides = NULL,
        .usage = VK_TENSOR_USAGE_SHADER_BIT_KHR,
    };

    VkTensorDescriptionKHR output_desc = input_desc;

    VkMLGraphCreateInfoKHR create_info = {
        .sType = (VkStructureType)VK_STRUCTURE_TYPE_ML_GRAPH_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .nodeCount = 1,
        .pNodes = NULL,
        .externalInputCount = 1,
        .pExternalInputDescriptions = &input_desc,
        .externalOutputCount = 1,
        .pExternalOutputDescriptions = &output_desc,
        .constantWeightCount = 0,
        .pConstantWeightDescriptions = NULL,
    };

    VkBool32 r = vk_ml_validate_graph_create(&create_info, &features, &props);
    expect("test_null_pnodes_with_nodecount", r, VK_FALSE);
}

int main(void)
{
    passed = 0;
    failed = 0;

    test_valid_single_node_graph();
    test_zero_node_count();
    test_exceed_max_nodes();
    test_zero_external_inputs();
    test_zero_external_outputs();
    test_cyclic_graph();
    test_self_reference();
    test_invalid_node_index();
    test_null_pnodes_with_nodecount();
    test_diamond_dag_valid();
    test_three_node_cycle();
    test_unknown_operation_stype();

    (void)printf("\nTotal: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
