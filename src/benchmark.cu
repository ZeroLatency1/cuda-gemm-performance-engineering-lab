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
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static fs::path project_results_dir() {
    if (const char* env = std::getenv("KUCH_RESULTS_DIR")) {
        if (*env) return fs::path(env);
    }
#if defined(__linux__)
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && !exe.empty()) {
        // CMake places the executable in <project>/build/.
        const fs::path candidate = exe.parent_path().parent_path() / "results";
        if (fs::exists(candidate.parent_path())) return candidate;
    }
#endif
    return fs::path("results");
}

static std::vector<float> make_input(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    uint32_t x = seed * 1664525u + 1013904223u;
    for (float& value : v) {
        x = 1664525u * x + 1013904223u;
        const float u = static_cast<float>((x >> 8) & 0x00ffffffu) / 16777216.0f;
        value = 2.0f * u - 1.0f;
    }
    return v;
}

static double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

static void launch_kernel(const BenchmarkConfig& cfg,
                          const float* dA, const float* dB, float* dC,
                          const __half* dAh, const __half* dBh) {
    if (cfg.dtype == "fp16") {
        if (cfg.kernel != "tensorcore") {
            throw std::runtime_error("fp16 currently supports kernel=tensorcore only");
        }
        if ((cfg.M % 16) || (cfg.N % 16) || (cfg.K % 16)) {
            throw std::runtime_error("tensorcore requires M, N, and K divisible by 16");
        }
        tensor_core_cuda_gemm(cfg.M, cfg.N, cfg.K, dAh, dBh, dC);
        CUDA_CHECK_LAST();
        return;
    }

    if (cfg.kernel == "naive") naive_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (cfg.kernel == "coalesced") coalesced_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (cfg.kernel == "shared16") shared_mem_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC, 16);
    else if (cfg.kernel == "shared32") shared_mem_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC, 32);
    else if (cfg.kernel == "register") register_tiled_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (cfg.kernel == "register64") register_tiled_cuda_gemm_64(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (cfg.kernel == "vectorized") {
        if ((cfg.N % 4) != 0) throw std::runtime_error("vectorized requires N divisible by 4");
        vectorized_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    } else if (cfg.kernel == "warp") warp_shuffle_cuda_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else if (cfg.kernel == "cublas") cublas_gemm(cfg.M, cfg.N, cfg.K, dA, dB, dC);
    else throw std::runtime_error("unsupported kernel: " + cfg.kernel);
    CUDA_CHECK_LAST();
}

static std::vector<std::string> kernels_for(const BenchmarkConfig& cfg) {
    if (cfg.dtype == "fp16") return {"tensorcore"};
    if (cfg.kernel != "all") return {cfg.kernel};
    return {"naive", "coalesced", "shared16", "shared32", "register", "register64", "vectorized", "warp", "cublas"};
}

static BenchmarkResult benchmark_one(const BenchmarkConfig& cfg, const std::string& kernel,
                                     const std::vector<float>& A, const std::vector<float>& B,
                                     const std::vector<float>& ref,
                                     const std::vector<__half>& Ah, const std::vector<__half>& Bh) {
    BenchmarkConfig run = cfg;
    run.kernel = kernel;

    const size_t a_count = static_cast<size_t>(cfg.M) * cfg.K;
    const size_t b_count = static_cast<size_t>(cfg.K) * cfg.N;
    const size_t c_count = static_cast<size_t>(cfg.M) * cfg.N;

    if (kernel == "cpu") {
        if (cfg.dtype != "fp32") throw std::runtime_error("cpu kernel currently supports fp32 only");
        std::vector<float> got(c_count, 0.0f);
        for (int i = 0; i < cfg.warmup; ++i) cpu_gemm(cfg.M, cfg.N, cfg.K, A.data(), B.data(), got.data());
        std::vector<double> samples;
        samples.reserve(cfg.iterations);
        for (int i = 0; i < cfg.iterations; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            cpu_gemm(cfg.M, cfg.N, cfg.K, A.data(), B.data(), got.data());
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        BenchmarkResult r;
        r.timestamp = utc_timestamp(); r.experiment_name = cfg.experiment_name;
        r.git_commit = current_git_commit(); r.gpu = gpu_name();
        r.compute_capability = gpu_compute_capability(); r.driver = driver_version();
        r.cuda_runtime = cuda_runtime_version_string(); r.kernel = kernel; r.dtype = cfg.dtype;
        r.M = cfg.M; r.N = cfg.N; r.K = cfg.K; r.warmup = cfg.warmup; r.iterations = cfg.iterations;
        r.median_ms = percentile(samples, 0.50); r.p95_ms = percentile(samples, 0.95);
        r.min_ms = *std::min_element(samples.begin(), samples.end());
        const double flops = 2.0 * static_cast<double>(cfg.M) * cfg.N * cfg.K;
        r.gflops = flops / (r.median_ms * 1e6);
        if (!cfg.verify) r.verification_status = "NOT_VERIFIED";
        else {
            const CorrectnessMetrics m = compare_outputs(ref, got);
            r.max_abs_err = m.max_abs; r.max_rel_err = m.max_rel;
            r.verification_status = m.pass ? "PASS" : "FAIL";
        }
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
    for (int i = 0; i < cfg.warmup; ++i) launch_kernel(run, dA, dB, dC, dAh, dBh);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(cfg.iterations);

    for (int i = 0; i < cfg.iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launch_kernel(run, dA, dB, dC, dAh, dBh);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(static_cast<double>(elapsed));
    }

    std::vector<float> got(c_count);
    CHECK_CUDA(cudaMemcpy(got.data(), dC, c_count * sizeof(float), cudaMemcpyDeviceToHost));

    BenchmarkResult r;
    r.timestamp = utc_timestamp();
    r.experiment_name = cfg.experiment_name;
    r.git_commit = current_git_commit();
    r.gpu = gpu_name();
    r.compute_capability = gpu_compute_capability();
    r.driver = driver_version();
    r.cuda_runtime = cuda_runtime_version_string();
    r.kernel = kernel;
    r.dtype = cfg.dtype;
    r.M = cfg.M; r.N = cfg.N; r.K = cfg.K;
    r.warmup = cfg.warmup; r.iterations = cfg.iterations;

    r.median_ms = percentile(samples, 0.50);
    r.p95_ms = percentile(samples, 0.95);
    r.min_ms = *std::min_element(samples.begin(), samples.end());
    const double flops = 2.0 * static_cast<double>(cfg.M) * cfg.N * cfg.K;
    r.gflops = flops / (r.median_ms * 1e6);

    if (!cfg.verify) {
        r.verification_status = "NOT_VERIFIED";
        r.notes = "verification disabled by --no-verify";
    } else {
        const CorrectnessMetrics m = compare_outputs(ref, got);
        r.max_abs_err = m.max_abs;
        r.max_rel_err = m.max_rel;
        r.verification_status = m.pass ? "PASS" : "FAIL";
        if (m.saw_nan) r.notes = "NaN detected";
        else if (m.saw_inf) r.notes = "Inf detected";
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dC));
    if (dA) CHECK_CUDA(cudaFree(dA));
    if (dB) CHECK_CUDA(cudaFree(dB));
    if (dAh) CHECK_CUDA(cudaFree(dAh));
    if (dBh) CHECK_CUDA(cudaFree(dBh));
    return r;
}

BenchmarkResult run_benchmark(const BenchmarkConfig& cfg) {
    if (cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0) throw std::runtime_error("dimensions must be positive");
    if (cfg.warmup < 0 || cfg.iterations <= 0) throw std::runtime_error("warmup must be >=0 and iterations >0");
    if (cfg.dtype != "fp32" && cfg.dtype != "fp16") throw std::runtime_error("dtype must be fp32 or fp16");

    const size_t a_count = static_cast<size_t>(cfg.M) * cfg.K;
    const size_t b_count = static_cast<size_t>(cfg.K) * cfg.N;
    const size_t c_count = static_cast<size_t>(cfg.M) * cfg.N;

    const auto A = make_input(a_count, static_cast<uint32_t>(cfg.seed));
    const auto B = make_input(b_count, static_cast<uint32_t>(cfg.seed + 1));
    std::vector<float> ref(c_count, 0.0f);
    if (cfg.verify) {
        if (cfg.M * 1LL * cfg.N * cfg.K > 3000000000LL) {
            throw std::runtime_error("CPU verification reference is intentionally blocked for very large GEMMs; use --no-verify or a smaller workload");
        }
        cpu_gemm(cfg.M, cfg.N, cfg.K, A.data(), B.data(), ref.data());
    }

    std::vector<__half> Ah;
    std::vector<__half> Bh;
    if (cfg.dtype == "fp16") {
        Ah.resize(a_count); Bh.resize(b_count);
        for (size_t i = 0; i < a_count; ++i) Ah[i] = __float2half(A[i]);
        for (size_t i = 0; i < b_count; ++i) Bh[i] = __float2half(B[i]);
        if (cfg.verify) {
            for (size_t i = 0; i < c_count; ++i) ref[i] = 0.0f;
            // Reference for FP16 input, FP32 accumulate.
            for (int i = 0; i < cfg.M; ++i)
                for (int k = 0; k < cfg.K; ++k) {
                    const float a = __half2float(Ah[static_cast<size_t>(i) * cfg.K + k]);
                    for (int j = 0; j < cfg.N; ++j)
                        ref[static_cast<size_t>(i) * cfg.N + j] += a * __half2float(Bh[static_cast<size_t>(k) * cfg.N + j]);
                }
        }
    }

    const auto kernels = kernels_for(cfg);
    BenchmarkResult first{};
    bool have_first = false;
    for (const std::string& k : kernels) {
        try {
            BenchmarkResult r = benchmark_one(cfg, k, A, B, ref, Ah, Bh);
            if (!have_first) { first = r; have_first = true; }
            std::cout << std::left << std::setw(16) << k
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(10) << r.median_ms
                      << std::setw(10) << r.p95_ms
                      << std::setw(10) << r.min_ms
                      << std::setw(14) << r.gflops
                      << "  " << std::setw(13) << r.verification_status
                      << "  abs=" << std::scientific << std::setprecision(3) << r.max_abs_err
                      << " rel=" << r.max_rel_err << '\n';
            std::cout << std::defaultfloat;
            append_experiment(r);
        } catch (const std::exception& e) {
            std::cerr << std::left << std::setw(16) << k << "UNSUPPORTED/ERROR: " << e.what() << '\n';
        }
    }
    if (!have_first) throw std::runtime_error("no runnable kernels for requested configuration");
    return first;
}


static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
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

static unsigned long long next_experiment_id(const fs::path& results_dir) {
    fs::create_directories(results_dir);
    const fs::path p = results_dir / "experiments.csv";
    unsigned long long max_id = 0;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("EXP-", 0) == 0) {
            try { max_id = std::max(max_id, std::stoull(line.substr(4))); } catch (...) {}
        }
    }
    return max_id + 1;
}

void append_experiment(BenchmarkResult result) {
    const fs::path results_dir = project_results_dir();
    fs::create_directories(results_dir / "raw");
    fs::create_directories(results_dir / "summaries");
    if (result.experiment_id.empty()) {
        std::ostringstream id;
        id << "EXP-" << std::setw(6) << std::setfill('0') << next_experiment_id(results_dir);
        result.experiment_id = id.str();
    }

    const fs::path csv = results_dir / "experiments.csv";
    const fs::path jsonl = results_dir / "experiments.jsonl";
    const bool header = !fs::exists(csv) || fs::file_size(csv) == 0;

    std::ofstream out(csv, std::ios::app);
    if (!out) throw std::runtime_error("cannot open results/experiments.csv for append");
    if (header) out << "experiment_id,timestamp,experiment_name,git_commit,GPU,compute_capability,driver,cuda_runtime,kernel,dtype,M,N,K,warmup,iterations,median_ms,p95_ms,min_ms,gflops,verification_status,max_abs_err,max_rel_err,notes\n";
    out << result.experiment_id << ','
        << json_escape(result.timestamp) << ','
        << json_escape(result.experiment_name) << ','
        << csv_escape(result.git_commit) << ','
        << csv_escape(result.gpu) << ','
        << csv_escape(result.compute_capability) << ','
        << csv_escape(result.driver) << ','
        << csv_escape(result.cuda_runtime) << ','
        << json_escape(result.kernel) << ','
        << json_escape(result.dtype) << ','
        << result.M << ',' << result.N << ',' << result.K << ','
        << result.warmup << ',' << result.iterations << ','
        << std::setprecision(12) << result.median_ms << ',' << result.p95_ms << ',' << result.min_ms << ','
        << result.gflops << ',' << result.verification_status << ','
        << result.max_abs_err << ',' << result.max_rel_err << ','
        << csv_escape(result.notes) << '\n';

    std::ofstream jout(jsonl, std::ios::app);
    if (!jout) throw std::runtime_error("cannot open results/experiments.jsonl for append");
    const std::string json_record =
        "{\"experiment_id\":\"" + json_escape(result.experiment_id) +
        "\",\"timestamp\":\"" + json_escape(result.timestamp) +
        "\",\"experiment_name\":\"" + json_escape(result.experiment_name) +
        "\",\"git_commit\":\"" + json_escape(result.git_commit) +
        "\",\"gpu\":\"" + json_escape(result.gpu) +
        "\",\"compute_capability\":\"" + json_escape(result.compute_capability) +
        "\",\"driver\":\"" + json_escape(result.driver) +
        "\",\"cuda_runtime\":\"" + json_escape(result.cuda_runtime) +
        "\",\"kernel\":\"" + json_escape(result.kernel) +
        "\",\"dtype\":\"" + json_escape(result.dtype) +
        "\",\"M\":" + std::to_string(result.M) +
        ",\"N\":" + std::to_string(result.N) +
        ",\"K\":" + std::to_string(result.K) +
        ",\"warmup\":" + std::to_string(result.warmup) +
        ",\"iterations\":" + std::to_string(result.iterations) +
        ",\"median_ms\":" + std::to_string(result.median_ms) +
        ",\"p95_ms\":" + std::to_string(result.p95_ms) +
        ",\"min_ms\":" + std::to_string(result.min_ms) +
        ",\"gflops\":" + std::to_string(result.gflops) +
        ",\"verification_status\":\"" + json_escape(result.verification_status) +
        "\",\"max_abs_err\":" + std::to_string(result.max_abs_err) +
        ",\"max_rel_err\":" + std::to_string(result.max_rel_err) +
        ",\"notes\":\"" + json_escape(result.notes) + "\"}";
    jout << json_record << '\n';

    std::ofstream raw(results_dir / "raw" / (result.experiment_id + ".json"), std::ios::trunc);
    if (!raw) throw std::runtime_error("cannot open per-experiment raw result for write");
    raw << json_record << '\n';

    std::cout << "Experiment recorded: " << result.experiment_id << " in " << results_dir << '\n';
}
