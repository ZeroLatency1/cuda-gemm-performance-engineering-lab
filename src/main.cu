#include "benchmark.h"
#include "gemm.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

static void print_help(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --M <int>                     Rows of A/C (default 1024)\n"
              << "  --N <int>                     Cols of B/C (default 1024)\n"
              << "  --K <int>                     Reduction dimension (default 1024)\n"
              << "  --kernel <str>                cpu, naive, coalesced, shared16, shared32, register, register64, vectorized, warp, tensorcore, cublas, all\n"
              << "  --dtype <str>                 fp32 or fp16 (default fp32)\n"
              << "  --warmup <int>                Warmup iterations (default 5)\n"
              << "  --iterations <int>            Timed iterations (default 10)\n"
              << "  --seed <int>                  Input seed (default 42)\n"
              << "  --verify | --no-verify        Enable/disable correctness verification\n"
              << "  --experiment-name <name>      Name stored in result history\n"
              << "  --help                        Show this help\n";
}

static BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (a == "--M") cfg.M = std::stoi(next("--M"));
        else if (a == "--N") cfg.N = std::stoi(next("--N"));
        else if (a == "--K") cfg.K = std::stoi(next("--K"));
        else if (a == "--kernel") cfg.kernel = next("--kernel");
        else if (a == "--dtype") cfg.dtype = next("--dtype");
        else if (a == "--warmup") cfg.warmup = std::stoi(next("--warmup"));
        else if (a == "--iterations") cfg.iterations = std::stoi(next("--iterations"));
        else if (a == "--seed") cfg.seed = std::stoi(next("--seed"));
        else if (a == "--verify") cfg.verify = true;
        else if (a == "--no-verify") cfg.verify = false;
        else if (a == "--experiment-name") cfg.experiment_name = next("--experiment-name");
        else if (a == "--help" || a == "-h") { print_help(argv[0]); return cfg; }
        else throw std::runtime_error("unknown argument: " + a);
    }
    return cfg;
}

int main(int argc, char** argv) {
    try {
        const BenchmarkConfig cfg = parse_args(argc, argv);
        if (cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0) throw std::runtime_error("dimensions must be positive");
        if (cfg.warmup < 0 || cfg.iterations <= 0) throw std::runtime_error("warmup must be >= 0 and iterations > 0");
        if (cfg.dtype != "fp32" && cfg.dtype != "fp16") throw std::runtime_error("dtype must be fp32 or fp16");

        int device_count = 0;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) throw std::runtime_error("no CUDA-capable device detected");

        std::cout << "CUDA GEMM Performance Engineering Lab\n"
                  << "====================================\n"
                  << "GPU: " << gpu_name() << " (compute " << gpu_compute_capability() << ")\n"
                  << "CUDA runtime API: " << cuda_runtime_version_string() << "\n"
                  << "Driver API: " << driver_version() << "\n"
                  << "Workload: M=" << cfg.M << " N=" << cfg.N << " K=" << cfg.K
                  << ", dtype=" << cfg.dtype << ", warmup=" << cfg.warmup
                  << ", iterations=" << cfg.iterations << "\n";
        run_benchmark(cfg);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
