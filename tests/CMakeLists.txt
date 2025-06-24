cmake_minimum_required(VERSION 3.20)
project(tinytorch-tests LANGUAGES CXX CUDA)

# Catch2 header is vendored under tests/catch2/.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/catch2)
include_directories(${CMAKE_SOURCE_DIR})

add_executable(test_tinytorch
    test_tinytorch.cpp
    ../tinytorch.cpp)

set_target_properties(test_tinytorch PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17)

enable_testing()
add_test(NAME tinytorch COMMAND test_tinytorch)
