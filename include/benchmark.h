#pragma once

#include <string>

struct BenchmarkConfig {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int warmup = 5;
    int iterations = 10;
    int seed = 42;
    bool verify = true;
    std::string kernel = "all";
    std::string dtype = "fp32";
    std::string experiment_name = "manual";
};

struct BenchmarkResult {
    std::string experiment_id;
    std::string timestamp;
    std::string experiment_name;
    std::string git_commit;
    std::string gpu;
    std::string compute_capability;
    std::string driver;
    std::string cuda_runtime;
    std::string kernel;
    std::string dtype;
    int M = 0;
    int N = 0;
    int K = 0;
    int warmup = 0;
    int iterations = 0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double min_ms = 0.0;
    double gflops = 0.0;
    std::string verification_status = "NOT_VERIFIED";
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    std::string notes;
};

BenchmarkResult run_benchmark(const BenchmarkConfig& cfg);
void append_experiment(BenchmarkResult result);
