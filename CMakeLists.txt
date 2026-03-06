cmake_minimum_required(VERSION 3.20)
project(vulkan_ml_primitives
    VERSION 0.1.0
    LANGUAGES C
    DESCRIPTION "VK_KHR_ml_primitives Vulkan extension reference implementation"
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan 1.3 REQUIRED)

# Compiler warning flags
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

option(BUILD_TESTING "Build tests and examples" ON)

# --------------------------------------------------------------------------
# Library: vk_ml_primitives (reference ICD layer implementation)
# --------------------------------------------------------------------------
set(ML_PRIMITIVES_SOURCES
    src/tensor.c
    src/tensor_view.c
    src/tensor_copy.c
    src/ml_primitives.c
    src/ml_graph.c
    src/ml_session.c
    src/ml_dispatch.c
    src/feature_query.c
)

add_library(vk_ml_primitives STATIC ${ML_PRIMITIVES_SOURCES})
target_include_directories(vk_ml_primitives
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vk_ml_primitives PUBLIC Vulkan::Vulkan $<$<NOT:$<C_COMPILER_ID:MSVC>>:m>)

# --------------------------------------------------------------------------
# Library: vk_ml_validation (validation layer)
# --------------------------------------------------------------------------
set(VALIDATION_SOURCES
    layers/validation/tensor_validation.c
    layers/validation/graph_validation.c
    layers/validation/session_validation.c
    layers/validation/dispatch_validation.c
    layers/validation/barrier_validation.c
)

add_library(vk_ml_validation STATIC ${VALIDATION_SOURCES})
target_include_directories(vk_ml_validation
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation
            ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vk_ml_validation PUBLIC Vulkan::Vulkan)

# --------------------------------------------------------------------------
# Install targets
# --------------------------------------------------------------------------
install(TARGETS vk_ml_primitives vk_ml_validation
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/vulkan
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(BUILD_TESTING)
enable_testing()

set(CTS_TESTS
    tests/cts/test_tensor_lifecycle.c
    tests/cts/test_tensor_view.c
    tests/cts/test_tensor_copy.c
    tests/cts/test_tensor_formats.c
    tests/cts/test_ml_graph.c
    tests/cts/test_ml_session.c
    tests/cts/test_ml_dispatch.c
    tests/cts/test_synchronization.c
    tests/cts/test_spirv_tensor.c
    tests/cts/test_oom.c
)

foreach(test_src ${CTS_TESTS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${test_name} PRIVATE vk_ml_primitives vk_ml_validation)
    add_test(NAME cts_${test_name} COMMAND ${test_name})
endforeach()

add_executable(test_vuids tests/validation/test_vuids.c)
target_include_directories(test_vuids PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation
)
target_link_libraries(test_vuids PRIVATE vk_ml_primitives vk_ml_validation)
add_test(NAME validation_test_vuids COMMAND test_vuids)

set(UNIT_TESTS
    tests/unit/test_dag_validation.c
    tests/unit/test_descriptor_validation.c
)

foreach(test_src ${UNIT_TESTS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation
    )
    target_link_libraries(${test_name} PRIVATE vk_ml_primitives vk_ml_validation)
    add_test(NAME unit_${test_name} COMMAND ${test_name})
endforeach()

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
add_executable(quickstart examples/quickstart.c)
target_include_directories(quickstart PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(quickstart PRIVATE vk_ml_primitives)

endif() # BUILD_TESTING

# --------------------------------------------------------------------------
# Static analysis (T003)
# --------------------------------------------------------------------------
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
    set(CMAKE_C_CLANG_TIDY
        ${CLANG_TIDY_EXE};
        -checks=-*,bugprone-*,cert-*,clang-analyzer-*,misc-*,performance-*,readability-*;
        -warnings-as-errors=*;
        -header-filter=${CMAKE_CURRENT_SOURCE_DIR}/include/.*
    )
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
else()
    message(WARNING "clang-tidy not found; static analysis disabled")
endif()

find_program(CPPCHECK_EXE NAMES cppcheck)
if(CPPCHECK_EXE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXE}
            --enable=all
            --suppress=missingIncludeSystem
            --error-exitcode=1
            --inline-suppr
            -I ${CMAKE_CURRENT_SOURCE_DIR}/include
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/layers
        COMMENT "Running cppcheck static analysis"
        VERBATIM
    )
    message(STATUS "cppcheck found: ${CPPCHECK_EXE}")
else()
    message(WARNING "cppcheck not found; cppcheck target unavailable")
endif()
