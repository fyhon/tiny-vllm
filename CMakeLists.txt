cmake_minimum_required(VERSION 3.24)

# Auto-detect nvcc: prefer CMAKE_CUDA_COMPILER env override, then PATH lookup,
# else fall back to common install locations. Original hardcoded "/opt/cuda" only
# matched the original author's setup; cuda-lab containers ship CUDA at /usr/local/cuda.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(NVCC_EXECUTABLE nvcc
        HINTS /usr/local/cuda/bin /opt/cuda/bin /usr/bin
        DOC "nvcc compiler path")
    if(NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${NVCC_EXECUTABLE}")
    endif()
endif()
if(NOT DEFINED CUDACXX)
    set(CUDACXX "${CMAKE_CUDA_COMPILER}")
endif()

project(tiny-vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

find_package(CUDAToolkit REQUIRED)

add_executable(tiny-vllm
    src/main.cpp
    src/kernels.cu
)

target_include_directories(tiny-vllm PRIVATE src)
target_include_directories(tiny-vllm PRIVATE include)

target_link_libraries(tiny-vllm PRIVATE
    CUDA::cublas
    CUDA::cudart
)

