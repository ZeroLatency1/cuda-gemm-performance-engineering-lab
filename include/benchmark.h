#pragma once

#include <limits>
#include <string>

struct BenchmarkConfig {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int warmup = 10;
    int iterations = 50;
    int seed = 42;
    bool verify = true;
    std::string kernel = "all";
    std::string dtype = "fp32";
    std::string experiment_name = "manual";
    std::string output_dir;
    std::string parent_experiment_id;
    std::string baseline_experiment_id;
    std::string optimization_description;
};

struct BenchmarkResult {
    std::string experiment_id;
    std::string timestamp;
    std::string hostname;
    std::string os;
    std::string wsl_status;
    std::string experiment_name;
    std::string git_commit;
    std::string gpu;
    std::string gpu_uuid;
    std::string compute_capability;
    std::string driver;
    std::string cuda_runtime;
    std::string cuda_toolkit;
    std::string compiler;
    std::string cmake_version;
    std::string kernel;
    std::string kernel_variant;
    std::string dtype;
    int M = 0;
    int N = 0;
    int K = 0;
    int warmup = 0;
    int iterations = 0;
    int seed = 0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double min_ms = 0.0;
    double gflops = 0.0;
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;
    double end_to_end_ms = 0.0;
    double end_to_end_gflops = 0.0;
    double latency_delta_ms = std::numeric_limits<double>::quiet_NaN();
    double latency_delta_pct = std::numeric_limits<double>::quiet_NaN();
    double throughput_delta_gflops = std::numeric_limits<double>::quiet_NaN();
    double throughput_delta_pct = std::numeric_limits<double>::quiet_NaN();
    std::string verification_status = "NOT_VERIFIED";
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    std::string status = "PASS";
    std::string cuda_errors;
    std::string runtime_errors;
    std::string environment_warnings;
    std::string parent_experiment_id;
    std::string baseline_experiment_id;
    std::string optimization_description;
    std::string notes;
};

BenchmarkResult run_benchmark(const BenchmarkConfig& cfg);
void append_experiment(BenchmarkResult result, const BenchmarkConfig& cfg);
