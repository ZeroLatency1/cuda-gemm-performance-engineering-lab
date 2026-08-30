cmake_minimum_required(VERSION 3.20)
project(cuda_gemm_performance_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "RTX 4060 Laptop SM architecture")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(CUDAToolkit 12.0 REQUIRED)

if (CUDAToolkit_VERSION VERSION_LESS 12.0 OR CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    message(FATAL_ERROR "This project targets CUDA 12.x; detected ${CUDAToolkit_VERSION}")
endif()

add_executable(cuda_gemm_benchmark
    src/main.cu
    src/benchmark.cu
    src/correctness.cu
    src/cpu_gemm.cpp
    src/naive_cuda.cu
    src/coalesced_cuda.cu
    src/shared_mem_cuda.cu
    src/register_tiled_cuda.cu
    src/vectorized_cuda.cu
    src/warp_experiments.cu
    src/tensor_core_cuda.cu
    src/cublas_gemm.cu
    src/utils.cu
)

target_include_directories(cuda_gemm_benchmark PRIVATE include)
target_link_libraries(cuda_gemm_benchmark PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(cuda_gemm_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(cuda_gemm_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89
)
