cmake_minimum_required(VERSION 3.20)
project(cuda_gemm_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

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
set_target_properties(cuda_gemm_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
