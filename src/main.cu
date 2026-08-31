#include "benchmark.h"
#include "gemm.h"
#include "utils.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void print_help(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n"
        << "  --M <int>                         Rows of A/C (default 1024)\n"
        << "  --N <int>                         Cols of B/C (default 1024)\n"
        << "  --K <int>                         Reduction dimension (default 1024)\n"
        << "  --kernel <str>                    cpu, naive, coalesced, shared16, shared32, register, register64, vectorized, warp, tensorcore, cublas, all\n"
        << "  --dtype <str>                     fp32 or fp16 (default fp32)\n"
        << "  --warmup <int>                    Warmup iterations (default 10)\n"
        << "  --iterations <int>                Timed iterations (default 50)\n"
        << "  --seed <int>                      Deterministic input seed (default 42)\n"
        << "  --verify | --no-verify            Enable/disable correctness verification\n"
        << "  --experiment-name <name>          Name stored in result history\n"
        << "  --output <dir>                    Explicit append-only result directory\n"
        << "  --parent-experiment-id <id>       Parent optimization experiment\n"
        << "  --baseline-experiment-id <id>     Baseline experiment for comparison\n"
        << "  --optimization-description <txt>  One-variable optimization description\n"
        << "  --allow-non-target-gpu            Explicitly permit execution on non-SM-8.9 GPUs\n"
        << "  --help                            Show this help\n";
}

struct ParseOutcome {
    BenchmarkConfig cfg;
    bool help = false;
    bool allow_non_target_gpu = false;
};

ParseOutcome parse_args(int argc, char** argv) {
    ParseOutcome out;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (a == "--M") out.cfg.M = std::stoi(next("--M"));
        else if (a == "--N") out.cfg.N = std::stoi(next("--N"));
        else if (a == "--K") out.cfg.K = std::stoi(next("--K"));
        else if (a == "--kernel") out.cfg.kernel = next("--kernel");
        else if (a == "--dtype") out.cfg.dtype = next("--dtype");
        else if (a == "--warmup") out.cfg.warmup = std::stoi(next("--warmup"));
        else if (a == "--iterations") out.cfg.iterations = std::stoi(next("--iterations"));
        else if (a == "--seed") out.cfg.seed = std::stoi(next("--seed"));
        else if (a == "--verify") out.cfg.verify = true;
        else if (a == "--no-verify") out.cfg.verify = false;
        else if (a == "--experiment-name") out.cfg.experiment_name = next("--experiment-name");
        else if (a == "--output") out.cfg.output_dir = next("--output");
        else if (a == "--parent-experiment-id") out.cfg.parent_experiment_id = next("--parent-experiment-id");
        else if (a == "--baseline-experiment-id") out.cfg.baseline_experiment_id = next("--baseline-experiment-id");
        else if (a == "--optimization-description") out.cfg.optimization_description = next("--optimization-description");
        else if (a == "--allow-non-target-gpu") out.allow_non_target_gpu = true;
        else if (a == "--help" || a == "-h") out.help = true;
        else throw std::runtime_error("unknown argument: " + a);
    }
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const ParseOutcome parsed = parse_args(argc, argv);
        if (parsed.help) { print_help(argv[0]); return EXIT_SUCCESS; }
        const BenchmarkConfig& cfg = parsed.cfg;

        if (cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0) throw std::runtime_error("dimensions must be positive");
        if (cfg.warmup < 0 || cfg.iterations <= 0) throw std::runtime_error("warmup must be >= 0 and iterations > 0");
        if (cfg.dtype != "fp32" && cfg.dtype != "fp16") throw std::runtime_error("dtype must be fp32 or fp16");

        int device_count = 0;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) throw std::runtime_error("no CUDA-capable device detected");
        if (!parsed.allow_non_target_gpu && gpu_compute_capability() != "8.9") {
            throw std::runtime_error("target fidelity check failed: detected compute capability " + gpu_compute_capability() + "; expected 8.9. Use --allow-non-target-gpu only for explicit non-target smoke tests.");
        }

        std::cout << "CUDA GEMM Performance Engineering Lab\n"
                  << "====================================\n"
                  << "GPU: " << gpu_name() << " (compute " << gpu_compute_capability() << ")\n"
                  << "GPU UUID: " << gpu_uuid() << "\n"
                  << "CUDA runtime/toolkit: " << cuda_runtime_version_string() << "/" << cuda_toolkit_version_string() << "\n"
                  << "Driver: " << driver_version() << "\n"
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
