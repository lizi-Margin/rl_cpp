cmake_minimum_required(VERSION 4.00.0)
project(rl_cpp
    LANGUAGES CXX CUDA  # 添加 CUDA 支持
)

# Project-wide properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)         # 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Cppcheck
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CPPCHECK_ARGS
        --enable=warning
        --std=c++14
        --force
        --verbose
        --quiet
        --inline-suppr
        --error-exitcode=1
        --language=c++
        -I ${CMAKE_CURRENT_LIST_DIR}/src 
        -I ${CMAKE_CURRENT_LIST_DIR}/include 
        ${CMAKE_CURRENT_LIST_DIR}/src
    )

    add_custom_target(
        check
        COMMAND cppcheck ${CPPCHECK_ARGS}
        COMMENT "Running Cppcheck"
    )
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)


find_package(spdlog REQUIRED)

# Dependencies
## PyTorch
if (NOT TORCH_FOUND)
    find_package(Torch REQUIRED)
    if (TORCH_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    endif()
endif (NOT TORCH_FOUND)

## CUDA 工具包
find_package(CUDAToolkit REQUIRED)  # 使用 CUDAToolkit（推荐新版）

# 定义 CUDA 库（包含 .cu 文件）
add_library(mlp_ac_cuda SHARED
    src/model/mlp_ac.cu  # 假设 .cu 文件路径为 src/model/mlp_ac.cu
)

# Define targets
add_executable(rl)

# Enable all warnings
target_compile_options(rl PRIVATE -Wall -pedantic)

# 包含目录（添加 CUDA 头文件路径）
set(INCLUDE_DIRS
    include
    src
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}  # 添加 CUDA 头文件路径
)
target_include_directories(rl PRIVATE ${INCLUDE_DIRS})
target_include_directories(mlp_ac_cuda PRIVATE ${INCLUDE_DIRS})

# 链接库（添加 CUDA 运行时库和自定义 CUDA 库）
target_link_libraries(rl PRIVATE
    mlp_ac_cuda
    torch 
    ${TORCH_LIBRARIES}
    CUDA::cudart  # 链接 CUDA 运行时库
    spdlog::spdlog
)
# Recurse into source tree
add_subdirectory(src)

# 以下为测试部分（可选）
# find_package(GTest REQUIRED)
# add_executable(rl_tests
#     tests/test_a2c.cpp
#     tests/test_grid.cpp
# )
# target_link_libraries(rl_tests
#     PRIVATE
#     rl
#     GTest::GTest
#     GTest::Main
# )