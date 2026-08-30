# Benchmark Analysis

No target-side numerical analysis is prefilled in this archive.

After `scripts/run_benchmarks.sh` has been executed on the RTX 4060 Laptop GPU, this document should be populated from `results/experiments.csv` and `results/experiments.jsonl` with:

- kernel-by-kernel median/p95/minimum latency
- GFLOPS from median kernel latency
- H2D/D2H/end-to-end medians
- workload-specific winner/regression analysis
- cuBLAS comparisons
- Tensor Core comparisons
- repeatability/variance observations

No fabricated values belong here.
