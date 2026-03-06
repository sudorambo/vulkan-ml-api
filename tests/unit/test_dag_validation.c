/**
 * @file test_dag_validation.c
 * @brief DAG validation unit tests (US2) - graph validation layer.
 */

#include <vulkan/vulkan_ml_primitives.h>
#include "../../src/internal.h"
#include "../../layers/validation/vk_ml_validation.h"
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

    (void)printf("\nTotal: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
