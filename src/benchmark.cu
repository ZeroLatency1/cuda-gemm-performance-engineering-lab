#include "benchmark.h"
#include "correctness.h"
#include "gemm.h"
#include "utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path project_root() {
#if defined(__linux__)
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && !exe.empty() && exe.parent_path().filename() == "build") return exe.parent_path().parent_path();
#endif
    return fs::current_path();
}

fs::path project_results_dir(const BenchmarkConfig& cfg) {
    if (!cfg.output_dir.empty()) return fs::path(cfg.output_dir);
    if (const char* env = std::getenv("KUCH_RESULTS_DIR")) {
        if (*env) return fs::path(env);
    }
    return project_root() / "results";
}

std::vector<float> make_input(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    uint32_t x = seed * 1664525u + 1013904223u;
    for (float& value : v) {
        x = 1664525u * x + 1013904223u;
        const float u = static_cast<float>((x >> 8) & 0x00ffffffu) / 16777216.0f;
        value = 2.0f * u - 1.0f;
    }
    return v;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::string csv_escape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

std::string num(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream out;
    out << std::setprecision(12) << x;
    return out.str();
}

unsigned long long max_experiment_id(const fs::path& results_dir) {
    unsigned long long max_id = 0;
    const fs::path csv = results_dir / "experiments.csv";
    std::ifstream in(csv);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find("EXP-");
        if (pos == std::string::npos) continue;
        try { max_id = std::max(max_id, std::stoull(line.substr(pos + 4))); } catch (...) {}
    }
    const fs::path raw_dir = results_dir / "raw";
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(raw_dir, ec)) {
        if (ec || !entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("EXP-", 0) != 0) continue;
        const size_t dot = name.find('.', 4);
        const std::string digits = name.substr(4, dot == std::string::npos ? std::string::npos : dot - 4);
        try { max_id = std::max(max_id, std::stoull(digits)); } catch (...) {}
    }
    return max_id;
}

std::string allocate_experiment_id(const fs::path& results_dir) {
    std::ostringstream id;
    id << "EXP-" << std::setw(6) << std::setfill('0') << (max_experiment_id(results_dir) + 1);
    return id.str();
}

void write_header_if_needed(const fs::path& csv) {
    constexpr const char* header =
        "experiment_id,timestamp,hostname,os,wsl_status,experiment_name,git_commit,gpu,gpu_uuid,compute_capability,driver,cuda_runtime,cuda_toolkit,compiler,cmake_version,kernel,kernel_variant,dtype,M,N,K,warmup,iterations,median_ms,p95_ms,min_ms,gflops,h2d_ms,d2h_ms,end_to_end_ms,end_to_end_gflops,verification_status,max_abs_error,max_rel_error,status,cuda_errors,runtime_errors,environment_warnings,parent_experiment_id,baseline_experiment_id,optimization_description,notes";
    if (!fs::exists(csv) || fs::file_size(csv) == 0) {
        std::ofstream out(csv, std::ios::app);
        if (!out) throw std::runtime_error("cannot open experiment CSV for append: " + csv.string());
        out << header << '\n';
        return;
    }
    std::ifstream in(csv);
    std::string existing;
    std::getline(in, existing);
    if (existing != header) {
        throw std::runtime_error("results/experiments.csv schema differs from the current append-only schema; historical data was not modified");
    }
}

void append_csv_record(const fs::path& csv, const BenchmarkResult& r) {
    std::ofstream out(csv, std::ios::app);
    if (!out) throw std::runtime_error("cannot open experiment CSV for append: " + csv.string());
    out << csv_escape(r.experiment_id) << ',' << csv_escape(r.timestamp) << ',' << csv_escape(r.hostname) << ','
        << csv_escape(r.os) << ',' << csv_escape(r.wsl_status) << ',' << csv_escape(r.experiment_name) << ','
        << csv_escape(r.git_commit) << ',' << csv_escape(r.gpu) << ',' << csv_escape(r.gpu_uuid) << ','
        << csv_escape(r.compute_capability) << ',' << csv_escape(r.driver) << ',' << csv_escape(r.cuda_runtime) << ','
        << csv_escape(r.cuda_toolkit) << ',' << csv_escape(r.compiler) << ',' << csv_escape(r.cmake_version) << ','
        << csv_escape(r.kernel) << ',' << csv_escape(r.kernel_variant) << ',' << csv_escape(r.dtype) << ','
        << r.M << ',' << r.N << ',' << r.K << ',' << r.warmup << ',' << r.iterations << ',' << r.seed << ','
        << num(r.median_ms) << ',' << num(r.p95_ms) << ',' << num(r.min_ms) << ',' << num(r.gflops) << ','
        << num(r.h2d_ms) << ',' << num(r.d2h_ms) << ',' << num(r.end_to_end_ms) << ',' << num(r.end_to_end_gflops) << ','
        << num(r.latency_delta_ms) << ',' << num(r.latency_delta_pct) << ',' << num(r.throughput_delta_gflops) << ',' << num(r.throughput_delta_pct) << ','
        << csv_escape(r.verification_status) << ',' << num(r.max_abs_error) << ',' << num(r.max_rel_error) << ','
        << csv_escape(r.status) << ',' << csv_escape(r.cuda_errors) << ',' << csv_escape(r.runtime_errors) << ','
        << csv_escape(r.environment_warnings) << ',' << csv_escape(r.parent_experiment_id) << ','
        << csv_escape(r.baseline_experiment_id) << ',' << csv_escape(r.optimization_description) << ',' << csv_escape(r.notes) << '\n';
}

std::string json_record(const BenchmarkResult& r) {
    std::ostringstream j;
    j << "{"
      << "\"experiment_id\":\"" << json_escape(r.experiment_id) << "\"," 
      << "\"timestamp\":\"" << json_escape(r.timestamp) << "\"," 
      << "\"hostname\":\"" << json_escape(r.hostname) << "\"," 
      << "\"os\":\"" << json_escape(r.os) << "\"," 
      << "\"wsl_status\":\"" << json_escape(r.wsl_status) << "\"," 
      << "\"experiment_name\":\"" << json_escape(r.experiment_name) << "\"," 
      << "\"git_commit\":\"" << json_escape(r.git_commit) << "\"," 
      << "\"gpu\":\"" << json_escape(r.gpu) << "\"," 
      << "\"gpu_uuid\":\"" << json_escape(r.gpu_uuid) << "\"," 
      << "\"compute_capability\":\"" << json_escape(r.compute_capability) << "\"," 
      << "\"driver\":\"" << json_escape(r.driver) << "\"," 
      << "\"cuda_runtime\":\"" << json_escape(r.cuda_runtime) << "\"," 
      << "\"cuda_toolkit\":\"" << json_escape(r.cuda_toolkit) << "\"," 
      << "\"compiler\":\"" << json_escape(r.compiler) << "\"," 
      << "\"cmake_version\":\"" << json_escape(r.cmake_version) << "\"," 
      << "\"kernel\":\"" << json_escape(r.kernel) << "\"," 
      << "\"kernel_variant\":\"" << json_escape(r.kernel_variant) << "\"," 
      << "\"dtype\":\"" << json_escape(r.dtype) << "\"," 
      << "\"M\":" << r.M << ",\"N\":" << r.N << ",\"K\":" << r.K
      << ",\"warmup\":" << r.warmup << ",\"iterations\":" << r.iterations << ",\"seed\":" << r.seed
      << ",\"median_ms\":" << num(r.median_ms) << ",\"p95_ms\":" << num(r.p95_ms) << ",\"min_ms\":" << num(r.min_ms)
      << ",\"gflops\":" << num(r.gflops) << ",\"h2d_ms\":" << num(r.h2d_ms) << ",\"d2h_ms\":" << num(r.d2h_ms)
      << ",\"end_to_end_ms\":" << num(r.end_to_end_ms) << ",\"end_to_end_gflops\":" << num(r.end_to_end_gflops)
      << ",\"latency_delta_ms\":" << num(r.latency_delta_ms) << ",\"latency_delta_pct\":" << num(r.latency_delta_pct)
      << ",\"throughput_delta_gflops\":" << num(r.throughput_delta_gflops) << ",\"throughput_delta_pct\":" << num(r.throughput_delta_pct)
      << ",\"verification_status\":\"" << json_escape(r.verification_status) << "\""
      << ",\"max_abs_error\":" << num(r.max_abs_error) << ",\"max_rel_error\":" << num(r.max_rel_error)
      << ",\"status\":\"" << json_escape(r.status) << "\""
      << ",\"cuda_errors\":\"" << json_escape(r.cuda_errors) << "\""
      << ",\"runtime_errors\":\"" << json_escape(r.runtime_errors) << "\""
      << ",\"environment_warnings\":\"" << json_escape(r.environment_warnings) << "\""
      << ",\"parent_experiment_id\":\"" << json_escape(r.parent_experiment_id) << "\""
      << ",\"baseline_experiment_id\":\"" << json_escape(r.baseline_experiment_id) << "\""
      << ",\"optimization_description\":\"" << json_escape(r.optimization_description) << "\""
      << ",\"notes\":\"" << json_escape(r.notes) << "\"}";
    return j.str();
}

BenchmarkResult base_result(const BenchmarkConfig& cfg, const std::string& kernel) {
    BenchmarkResult r;
    r.timestamp = utc_timestamp();
    r.hostname = hostname_string();
    r.os = os_description();
    r.wsl_status = wsl_status();
    r.experiment_name = cfg.experiment_name;
    r.git_commit = current_git_commit();
    r.gpu = gpu_name();
    r.gpu_uuid = gpu_uuid();
    r.compute_capability = gpu_compute_capability();
    r.driver = driver_version();
    r.cuda_runtime = cuda_runtime_version_string();
    r.cuda_toolkit = cuda_toolkit_version_string();
    r.compiler = host_compiler_version_string();
    r.cmake_version = cmake_version_string();
    r.kernel = kernel;
    if (kernel == "shared16") r.kernel_variant = "shared_tile_16x16";
    else if (kernel == "shared32") r.kernel_variant = "shared_tile_32x32";
    else if (kernel == "register") r.kernel_variant = "register_tile_32x32_bk8_tm2tn2";
    else if (kernel == "register64") r.kernel_variant = "register_tile_64x64_bk8_tm4tn4";
    else if (kernel == "vectorized") r.kernel_variant = "float4_output";
    else if (kernel == "tensorcore") r.kernel_variant = "wmma_16x16x16";
    else if (kernel == "cublas") r.kernel_variant = cfg.dtype == "fp16" ? "cublas_gemmex_tensorop" : "cublas_sgemm";
    else r.kernel_variant = kernel;
    r.dtype = cfg.dtype;
    r.M = cfg.M; r.N = cfg.N; r.K = cfg.K;
    r.warmup = cfg.warmup; r.iterations = cfg.iterations; r.seed = cfg.seed;
    r.parent_experiment_id = cfg.parent_experiment_id;
    r.baseline_experiment_id = cfg.baseline_experiment_id;
    r.optimization_description = cfg.optimization_description;
    r.environment_warnings = environment_warnings();
    return r;
}

bool supported(const BenchmarkConfig& cfg, const std::string& kernel, std::string& reason) {
    if (kernel == "cpu") {
        if (cfg.dtype != "fp32") { reason = "CPU reference benchmark currently accepts dtype=fp32 only"; return false; }
        return true;
    }
    if (kernel == "tensorcore") {
        if (cfg.dtype != "fp16") { reason = "WMMA Tensor Core path is exposed for dtype=fp16"; return false; }
        if ((cfg.M % 16) || (cfg.N % 16) || (cfg.K % 16)) { reason = "WMMA requires M, N, and K divisible by 16"; return false; }
        return true;
    }
    if (kernel == "cublas") return true;
    if (cfg.dtype == "fp16") { reason = "custom non-Tensor-Core kernels are fp32 implementations"; return false; }
    if (kernel == "vectorized" && cfg.N % 4 != 0) { reason = "float4 vectorized kernel requires N divisible by 4"; return false; }
    const std::vector<std::string> names = {"naive","coalesced","shared16","shared32","register","register64","vectorized","warp"};
    if (std::find(names.begin(), names.end(), kernel) == names.end()) { reason = "unknown kernel"; return false; }
    return true;
}

void launch_kernel(const BenchmarkConfig& cfg,
                   const std::string& kernel,
                   const float* dA, const float* dB, float* dC,
                   const __half* dAh, const __half* dBh) {
    if (kernel == "naive") naive_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (kernel == "coalesced") coalesced_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (kernel == "shared16") shared_mem_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC, 16);
    else if (kernel == "shared32") shared_mem_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC, 32);
    else if (kernel == "register") register_tiled_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (kernel == "register64") register_tiled_cuda_gemm_64(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (kernel == "vectorized") vectorized_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (kernel == "warp") warp_shuffle_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (kernel == "cublas") {
        if (cfg.dtype == "fp16") cublas_gemm_half(cfg.M, cfg.N, cfg.K, dAh, dBh, dC);
        else cublas_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    } else if (kernel == "tensorcore") tensor_core_cuda_gemm(cfg.M, cfg.N, cfg.K, dAh, dBh, dC);
    else throw std::runtime_error("unsupported kernel name: " + kernel);
    CUDA_CHECK_LAST();
}

std::vector<std::string> kernels_for(const BenchmarkConfig& cfg) {
    if (cfg.kernel != "all") return {cfg.kernel};
    if (cfg.dtype == "fp16") return {"tensorcore", "cublas"};
    return {"naive", "coalesced", "shared16", "shared32", "register", "register64", "vectorized", "warp", "cublas"};
}

void measure_transfers_fp32(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                            float* dA, float* dB, float* dC, int iterations,
                            double& h2d_ms, double& d2h_ms, double& end_to_end_ms,
                            const BenchmarkConfig& cfg, const std::string& kernel) {
    std::vector<double> h2d, d2h, e2e;
    h2d.reserve(iterations); d2h.reserve(iterations); e2e.reserve(iterations);
    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); h2d.push_back(ms);

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); d2h.push_back(ms);

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
        launch_kernel(cfg, kernel, dA, dB, dC, nullptr, nullptr);
        CHECK_CUDA(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); e2e.push_back(ms);
    }
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    h2d_ms = percentile(h2d, 0.50); d2h_ms = percentile(d2h, 0.50); end_to_end_ms = percentile(e2e, 0.50);
}

void measure_transfers_fp16(const std::vector<__half>& A, const std::vector<__half>& B, std::vector<float>& C,
                            __half* dA, __half* dB, float* dC, int iterations,
                            double& h2d_ms, double& d2h_ms, double& end_to_end_ms,
                            const BenchmarkConfig& cfg, const std::string& kernel) {
    std::vector<double> h2d, d2h, e2e;
    h2d.reserve(iterations); d2h.reserve(iterations); e2e.reserve(iterations);
    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); h2d.push_back(ms);

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); d2h.push_back(ms);

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(__half), cudaMemcpyHostToDevice));
        launch_kernel(cfg, kernel, nullptr, nullptr, dC, dA, dB);
        CHECK_CUDA(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); e2e.push_back(ms);
    }
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    h2d_ms = percentile(h2d, 0.50); d2h_ms = percentile(d2h, 0.50); end_to_end_ms = percentile(e2e, 0.50);
}

BenchmarkResult benchmark_one(const BenchmarkConfig& cfg, const std::string& kernel,
                              const std::vector<float>& A, const std::vector<float>& B,
                              const std::vector<float>& ref,
                              const std::vector<__half>& Ah, const std::vector<__half>& Bh) {
    BenchmarkResult r = base_result(cfg, kernel);
    const size_t a_count = static_cast<size_t>(cfg.M) * cfg.K;
    const size_t b_count = static_cast<size_t>(cfg.K) * cfg.N;
    const size_t c_count = static_cast<size_t>(cfg.M) * cfg.N;
    const double flops = 2.0 * static_cast<double>(cfg.M) * cfg.N * cfg.K;

    if (kernel == "cpu") {
        std::vector<float> got(c_count, 0.0f);
        std::vector<double> samples;
        samples.reserve(cfg.iterations);
        for (int i = 0; i < cfg.warmup; ++i) cpu_gemm(cfg.M, cfg.N, cfg.K, A.data(), B.data(), got.data());
        for (int i = 0; i < cfg.iterations; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            cpu_gemm(cfg.M, cfg.N, cfg.K, A.data(), B.data(), got.data());
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        r.median_ms = percentile(samples, 0.50); r.p95_ms = percentile(samples, 0.95); r.min_ms = *std::min_element(samples.begin(), samples.end());
        r.gflops = flops / (r.median_ms * 1e6);
        r.verification_status = cfg.verify ? (compare_outputs(ref, got).pass ? "PASS" : "FAIL") : "NOT_VERIFIED";
        if (cfg.verify) {
            const auto m = compare_outputs(ref, got); r.max_abs_error = m.max_abs; r.max_rel_error = m.max_rel;
            r.status = m.pass ? "PASS" : "ERROR";
        }
        r.notes = "CPU wall-clock timing; GPU timing uses CUDA events";
        return r;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    __half *dAh = nullptr, *dBh = nullptr;
    CHECK_CUDA(cudaMalloc(&dC, c_count * sizeof(float)));
    if (cfg.dtype == "fp16") {
        CHECK_CUDA(cudaMalloc(&dAh, a_count * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&dBh, b_count * sizeof(__half)));
        CHECK_CUDA(cudaMemcpy(dAh, Ah.data(), a_count * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dBh, Bh.data(), b_count * sizeof(__half), cudaMemcpyHostToDevice));
    } else {
        CHECK_CUDA(cudaMalloc(&dA, a_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dB, b_count * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dA, A.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMemset(dC, 0, c_count * sizeof(float)));

    for (int i = 0; i < cfg.warmup; ++i) launch_kernel(cfg, kernel, dA, dB, dC, dAh, dBh);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    std::vector<double> samples; samples.reserve(cfg.iterations);
    for (int i = 0; i < cfg.iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launch_kernel(cfg, kernel, dA, dB, dC, dAh, dBh);
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); samples.push_back(ms);
    }
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));

    std::vector<float> got(c_count);
    CHECK_CUDA(cudaMemcpy(got.data(), dC, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    r.median_ms = percentile(samples, 0.50); r.p95_ms = percentile(samples, 0.95); r.min_ms = *std::min_element(samples.begin(), samples.end());
    r.gflops = flops / (r.median_ms * 1e6);

    if (cfg.verify) {
        const auto m = compare_outputs(ref, got);
        r.max_abs_error = m.max_abs; r.max_rel_error = m.max_rel;
        r.verification_status = m.pass ? "PASS" : "FAIL";
        if (m.saw_nan) r.notes = "NaN detected";
        else if (m.saw_inf) r.notes = "Inf detected";
        if (!m.pass) r.status = "ERROR";
    } else {
        r.verification_status = "NOT_VERIFIED";
        r.notes = "verification disabled by --no-verify";
    }

    // These are separate from kernel-only timing and are never used for kernel GFLOPS.
    if (cfg.dtype == "fp16") measure_transfers_fp16(Ah, Bh, got, dAh, dBh, dC, std::max(1, std::min(cfg.iterations, 20)), r.h2d_ms, r.d2h_ms, r.end_to_end_ms, cfg, kernel);
    else measure_transfers_fp32(A, B, got, dA, dB, dC, std::max(1, std::min(cfg.iterations, 20)), r.h2d_ms, r.d2h_ms, r.end_to_end_ms, cfg, kernel);
    r.end_to_end_gflops = flops / (r.end_to_end_ms * 1e6);

    if (dA) CHECK_CUDA(cudaFree(dA)); if (dB) CHECK_CUDA(cudaFree(dB)); if (dAh) CHECK_CUDA(cudaFree(dAh)); if (dBh) CHECK_CUDA(cudaFree(dBh)); CHECK_CUDA(cudaFree(dC));
    return r;
}

} // namespace


struct BaselineRecord {
    bool found = false;
    std::string dtype;
    int M = 0, N = 0, K = 0;
    double median_ms = 0.0;
    double gflops = 0.0;
};

BaselineRecord find_baseline_record(const fs::path& jsonl, const std::string& id) {
    BaselineRecord out;
    if (id.empty()) return out;
    std::ifstream in(jsonl);
    std::string line;
    const std::string needle = "\"experiment_id\":\"" + json_escape(id) + "\"";
    while (std::getline(in, line)) {
        if (line.find(needle) == std::string::npos) continue;
        std::smatch m;
        if (std::regex_search(line, m, std::regex("\\\"dtype\\\":\\\"([^\\\"]+)\\\""))) out.dtype = m[1].str();
        if (std::regex_search(line, m, std::regex("\\\"M\\\":([0-9]+)"))) out.M = std::stoi(m[1].str());
        if (std::regex_search(line, m, std::regex("\\\"N\\\":([0-9]+)"))) out.N = std::stoi(m[1].str());
        if (std::regex_search(line, m, std::regex("\\\"K\\\":([0-9]+)"))) out.K = std::stoi(m[1].str());
        if (std::regex_search(line, m, std::regex("\\\"median_ms\\\":([-+0-9.eE]+)"))) out.median_ms = std::stod(m[1].str());
        if (std::regex_search(line, m, std::regex("\\\"gflops\\\":([-+0-9.eE]+)"))) out.gflops = std::stod(m[1].str());
        out.found = true;
        break;
    }
    return out;
}

void append_experiment(BenchmarkResult result, const BenchmarkConfig& cfg) {
    const fs::path dir = project_results_dir(cfg);
    fs::create_directories(dir / "raw");
    fs::create_directories(dir / "summaries");
    write_header_if_needed(dir / "experiments.csv");
    if (result.experiment_id.empty()) result.experiment_id = allocate_experiment_id(dir);
    const fs::path raw = dir / "raw" / (result.experiment_id + ".json");
    if (fs::exists(raw)) throw std::runtime_error("refusing to overwrite existing raw experiment file: " + raw.string());

    if (!result.baseline_experiment_id.empty()) {
        const BaselineRecord b = find_baseline_record(dir / "experiments.jsonl", result.baseline_experiment_id);
        if (!b.found) {
            result.notes += (result.notes.empty() ? "" : "; ") + std::string("baseline experiment not found; deltas unavailable");
        } else if (b.dtype != result.dtype || b.M != result.M || b.N != result.N || b.K != result.K) {
            result.notes += (result.notes.empty() ? "" : "; ") + std::string("baseline dimensions/dtype do not match; deltas unavailable");
        } else if (b.median_ms > 0.0 && b.gflops > 0.0) {
            result.latency_delta_ms = result.median_ms - b.median_ms;
            result.latency_delta_pct = (result.latency_delta_ms / b.median_ms) * 100.0;
            result.throughput_delta_gflops = result.gflops - b.gflops;
            result.throughput_delta_pct = (result.throughput_delta_gflops / b.gflops) * 100.0;
        }
    }

    const std::string json = json_record(result);
    append_csv_record(dir / "experiments.csv", result);
    {
        std::ofstream out(dir / "experiments.jsonl", std::ios::app);
        if (!out) throw std::runtime_error("cannot open experiments.jsonl for append");
        out << json << '\n';
    }
    {
        std::ofstream out(raw);
        if (!out) throw std::runtime_error("cannot create raw experiment file: " + raw.string());
        out << json << '\n';
    }
    std::cout << "Experiment recorded: " << result.experiment_id << " [" << result.status << "]\n";
}

BenchmarkResult run_benchmark(const BenchmarkConfig& cfg) {
    if (cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0) throw std::runtime_error("dimensions must be positive");
    if (cfg.warmup < 0 || cfg.iterations <= 0) throw std::runtime_error("warmup must be >= 0 and iterations > 0");
    if (cfg.dtype != "fp32" && cfg.dtype != "fp16") throw std::runtime_error("dtype must be fp32 or fp16");

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) throw std::runtime_error("no CUDA-capable device detected");

    const size_t a_count = static_cast<size_t>(cfg.M) * cfg.K;
    const size_t b_count = static_cast<size_t>(cfg.K) * cfg.N;
    const size_t c_count = static_cast<size_t>(cfg.M) * cfg.N;
    const auto A = make_input(a_count, static_cast<uint32_t>(cfg.seed));
    const auto B = make_input(b_count, static_cast<uint32_t>(cfg.seed + 1));

    std::vector<float> ref(c_count, 0.0f);
    if (cfg.verify) {
        if (static_cast<long double>(cfg.M) * cfg.N * cfg.K > 3.0e9L) {
            throw std::runtime_error("CPU verification reference is intentionally blocked for this very large workload; use --no-verify for performance-only runs");
        }
        cpu_gemm(cfg.M, cfg.N, cfg.K, A.data(), B.data(), ref.data());
    }

    std::vector<__half> Ah, Bh;
    if (cfg.dtype == "fp16") {
        Ah.resize(a_count); Bh.resize(b_count);
        for (size_t i = 0; i < a_count; ++i) Ah[i] = __float2half(A[i]);
        for (size_t i = 0; i < b_count; ++i) Bh[i] = __float2half(B[i]);
        if (cfg.verify) {
            std::fill(ref.begin(), ref.end(), 0.0f);
            for (int i = 0; i < cfg.M; ++i)
                for (int k = 0; k < cfg.K; ++k) {
                    const float a = __half2float(Ah[static_cast<size_t>(i) * cfg.K + k]);
                    for (int j = 0; j < cfg.N; ++j) ref[static_cast<size_t>(i) * cfg.N + j] += a * __half2float(Bh[static_cast<size_t>(k) * cfg.N + j]);
                }
        }
    }

    std::cout << "CUDA GEMM Performance Engineering Lab\n"
              << "GPU: " << gpu_name() << " (SM " << gpu_compute_capability() << ")\n"
              << "CUDA runtime/toolkit: " << cuda_runtime_version_string() << "/" << cuda_toolkit_version_string() << "\n"
              << "Driver: " << driver_version() << "\n"
              << "Workload: M=" << cfg.M << " N=" << cfg.N << " K=" << cfg.K << ", dtype=" << cfg.dtype
              << ", warmup=" << cfg.warmup << ", iterations=" << cfg.iterations << '\n';

    const auto kernels = kernels_for(cfg);
    BenchmarkResult first{};
    bool got_success = false;
    bool hard_error = false;
    bool correctness_error = false;

    for (const std::string& kernel : kernels) {
        std::string reason;
        if (!supported(cfg, kernel, reason)) {
            BenchmarkResult r = base_result(cfg, kernel);
            r.status = "UNSUPPORTED";
            r.verification_status = "NOT_APPLICABLE";
            r.notes = reason;
            try { append_experiment(r, cfg); } catch (const std::exception& e) { throw std::runtime_error(std::string("cannot log unsupported configuration: ") + e.what()); }
            std::cout << std::left << std::setw(16) << kernel << "UNSUPPORTED: " << reason << '\n';
            continue;
        }

        try {
            BenchmarkResult r = benchmark_one(cfg, kernel, A, B, ref, Ah, Bh);
            append_experiment(r, cfg);
            std::cout << std::left << std::setw(16) << kernel
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(12) << r.median_ms
                      << std::setw(12) << r.p95_ms
                      << std::setw(12) << r.min_ms
                      << std::setw(14) << r.gflops
                      << std::setw(12) << r.h2d_ms
                      << std::setw(12) << r.d2h_ms
                      << std::setw(14) << r.end_to_end_ms
                      << "  " << std::setw(13) << r.verification_status
                      << "  abs=" << std::scientific << std::setprecision(3) << r.max_abs_error
                      << " rel=" << r.max_rel_error << '\n' << std::defaultfloat;
            if (!got_success) { first = r; got_success = true; }
            if (r.status == "ERROR") {
                hard_error = true;
                if (cfg.verify && r.verification_status == "FAIL") correctness_error = true;
            }
        } catch (const std::exception& e) {
            BenchmarkResult r = base_result(cfg, kernel);
            r.status = "ERROR";
            r.verification_status = "NOT_VERIFIED";
            if (std::string(e.what()).find("CUDA error") != std::string::npos) r.cuda_errors = e.what();
            else r.runtime_errors = e.what();
            r.notes = "kernel execution failed; no performance claim recorded";
            append_experiment(r, cfg);
            std::cerr << std::left << std::setw(16) << kernel << "ERROR: " << e.what() << '\n';
            hard_error = true;
        }
    }

    if (!got_success) throw std::runtime_error("no runnable kernels for requested configuration");
    if (correctness_error) throw std::runtime_error("one or more verified kernels failed correctness");
    if (hard_error) throw std::runtime_error("one or more requested kernels failed; see experiment history for ERROR records");
    return first;
}
