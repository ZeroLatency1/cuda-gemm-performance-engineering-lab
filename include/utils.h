#pragma once

#include <cuda_runtime.h>
#include <string>

void check_cuda(cudaError_t status, const char* expression, const char* file, int line);

#define CHECK_CUDA(call) check_cuda((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() CHECK_CUDA(cudaGetLastError())

void randomize_matrix(float* mat, int size, int seed);
void randomize_matrix_half(void* mat, int size, int seed);

std::string utc_timestamp();
std::string current_git_commit();
std::string hostname_string();
std::string gpu_name();
std::string gpu_uuid();
std::string gpu_compute_capability();
std::string driver_version();
std::string cuda_runtime_version_string();
std::string cuda_toolkit_version_string();
std::string host_compiler_version_string();
std::string cmake_version_string();
std::string os_description();
std::string wsl_status();
std::string environment_warnings();
