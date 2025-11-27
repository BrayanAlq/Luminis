cmake_minimum_required(VERSION 3.18)

project(LuminisTests LANGUAGES CXX CUDA)

# Enable testing
enable_testing()

# Find GTest
find_package(GTest REQUIRED)

# Add test executable
add_executable(run_tests
    test_main.cpp
    test_filesystem.cpp
    test_preprocesamiento.cpp
    test_procesamiento_gpu.cu
    ${CMAKE_SOURCE_DIR}/filesystem_utils.cpp
    ${CMAKE_SOURCE_DIR}/preprocesamiento.cpp
    ${CMAKE_SOURCE_DIR}/procesamiento_gpu.cu
)

# Set CUDA properties
set_target_properties(run_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

# Link GTest
target_link_libraries(run_tests
    PRIVATE
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

# Include directories
target_include_directories(run_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR} # Include the parent directory
)

# Add test
add_test(NAME run_tests COMMAND run_tests)
