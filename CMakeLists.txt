cmake_minimum_required(VERSION 4.00.0)
project(rl_cpp
    LANGUAGES CXX
)

# Project-wide properties
set(CMAKE_CXX_STANDARD 17)

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

# Dependencies
## PyTorch
if (NOT TORCH_FOUND)
    find_package(Torch REQUIRED)
    if (TORCH_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
    endif()
endif (NOT TORCH_FOUND)

# Define targets
add_executable(rl)

# Enable all warnings
target_compile_options(rl PRIVATE -Wall -pedantic)

# Includes
set(INCLUDE_DIRS
    include
    src
    ${TORCH_INCLUDE_DIRS}
)
target_include_directories(rl PRIVATE ${INCLUDE_DIRS})

# Linking
target_link_libraries(rl torch ${TORCH_LIBRARIES})

# Recurse into source tree
add_subdirectory(src)

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