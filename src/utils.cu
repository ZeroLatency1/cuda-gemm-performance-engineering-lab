#include "utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ':' << line
            << " for " << expression << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

void randomize_matrix(float* mat, int size, int seed) {
    std::mt19937 gen(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) mat[i] = dis(gen);
}

void randomize_matrix_half(void* mat, int size, int seed) {
    auto* h_mat = static_cast<__half*>(mat);
    std::mt19937 gen(static_cast<unsigned>(seed));
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) h_mat[i] = __float2half(dis(gen));
}

static std::string command_output(const char* command) {
    std::array<char, 256> buffer{};
    std::string output;
    FILE* pipe = popen(command, "r");
    if (!pipe) return "UNKNOWN";
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) output += buffer.data();
    pclose(pipe);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r' || output.back() == ' ' || output.back() == '\t')) output.pop_back();
    return output.empty() ? "UNKNOWN" : output;
}

static std::string first_line(const std::string& s) {
    const auto pos = s.find('\n');
    return pos == std::string::npos ? s : s.substr(0, pos);
}

std::string utc_timestamp() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t tt = clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string hostname_string() {
    char host[256]{};
    return gethostname(host, sizeof(host) - 1) == 0 ? std::string(host) : "UNKNOWN";
}

std::string current_git_commit() {
    return command_output("git rev-parse --verify HEAD 2>/dev/null");
}

std::string gpu_name() {
    cudaDeviceProp prop{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return "UNKNOWN";
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "UNKNOWN";
    return prop.name;
}

std::string gpu_uuid() {
    cudaDeviceProp prop{};
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess) return "UNKNOWN";
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "UNKNOWN";

    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');

    for (unsigned char c : prop.uuid.bytes) {
        out << std::setw(2) << static_cast<unsigned>(c);
    }

    return out.str();
}

std::string gpu_compute_capability() {
    cudaDeviceProp prop{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return "UNKNOWN";
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "UNKNOWN";
    return std::to_string(prop.major) + "." + std::to_string(prop.minor);
}

std::string driver_version() {
    const std::string nvidia_smi =
        command_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null");

    if (nvidia_smi != "UNKNOWN" && !nvidia_smi.empty()) {
        return first_line(nvidia_smi);
    }

    int version = 0;
    if (cudaDriverGetVersion(&version) != cudaSuccess) return "UNKNOWN";

    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

std::string cuda_runtime_version_string() {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess) return "UNKNOWN";
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

std::string cuda_toolkit_version_string() {
    return std::to_string(CUDART_VERSION / 1000) + "." + std::to_string((CUDART_VERSION % 1000) / 10);
}

std::string host_compiler_version_string() {
    const std::string version =
        command_output("gcc -dumpfullversion -dumpversion 2>/dev/null");

    if (version != "UNKNOWN" && !version.empty()) {
        return first_line(version);
    }

    return "UNKNOWN";
}

std::string cmake_version_string() {
    return first_line(command_output("cmake --version 2>/dev/null"));
}

std::string os_description() {
    std::ifstream f("/etc/os-release");
    std::string line;
    std::string pretty = "UNKNOWN";
    while (std::getline(f, line)) {
        if (line.rfind("PRETTY_NAME=", 0) == 0) {
            pretty = line.substr(12);
            if (pretty.size() >= 2 && pretty.front() == '"' && pretty.back() == '"') pretty = pretty.substr(1, pretty.size() - 2);
            break;
        }
    }
    return pretty;
}

std::string wsl_status() {
    if (access("/dev/dxg", F_OK) == 0) return "WSL2_GPU_DEVICE_PRESENT";
    const std::string proc = command_output("grep -qi microsoft /proc/version && echo WSL_DETECTED || echo NOT_WSL");
    return proc == "WSL_DETECTED" ? "WSL_DETECTED_NO_DXG" : "NOT_WSL";
}

std::string environment_warnings() {
    std::vector<std::string> warnings;
    const std::string os = os_description();
    if (os.find("Ubuntu 24.04") == std::string::npos) warnings.push_back("OS differs from target Ubuntu 24.04");
    const std::string cc = gpu_compute_capability();
    if (cc != "8.9") warnings.push_back("GPU compute capability differs from target 8.9");
    if (cuda_toolkit_version_string().rfind("12.0", 0) != 0) warnings.push_back("CUDA toolkit differs from target 12.0.x");
    if (host_compiler_version_string().rfind("13.", 0) != 0) warnings.push_back("host compiler differs from target GCC 13.x");
    return warnings.empty() ? "NONE" : [&] {
        std::ostringstream out;
        for (size_t i = 0; i < warnings.size(); ++i) {
            if (i) out << "; ";
            out << warnings[i];
        }
        return out.str();
    }();
}
