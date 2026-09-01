# Profiling Methodology

## Nsight Compute

Use Nsight Compute to answer a specific kernel hypothesis: register pressure, memory throughput, occupancy, instruction issue, cache behavior, or whether Tensor Core instructions are present.

Run a focused profile with:

```bash
./scripts/run_profile.sh --kernel register --M 2048 --N 2048 --K 2048 --dtype fp32
```

The profiling script discovers `ncu` from `PATH` or known Linux installation locations. It captures the benchmark's generated `EXP-XXXXXX` identifier, moves the resulting report into `profiles/EXP-XXXXXX/profile.ncu-rep`, writes a human-readable log and JSON metadata beside it, and appends a link to `results/profiling.jsonl`.

Profiles are diagnostic artifacts, not performance benchmarks. Nsight Compute may run multiple metric passes and materially perturb kernel timing, so `performance_claim_eligible` is always `false` for profile records. Use `results/experiments.csv`/`.jsonl` and ordinary benchmark runs for performance claims.

The same experiment lineage flags can be passed to the profile script when appropriate: `--parent-experiment-id`, `--baseline-experiment-id`, and `--optimization-description`. The profile script records those lineage values in the profiling metadata, but deliberately does **not** pass `--baseline-experiment-id` into the benchmark invocation, so Nsight-instrumented timings cannot produce misleading speedup/regression deltas. The profiled invocation remains a distinct experiment so that its environment and command are reproducible.

## Nsight Systems

Nsight Systems is optional in this target environment. It is appropriate for system-level timeline questions such as transfer overlap, launch overhead, CPU/GPU concurrency, and synchronization behavior.

## Failure handling

A profiler failure is not converted into a fake success. The profile script stores the timestamp, tool, version, exact command, workload, exit code, and any observed experiment ID under `profiles/limitations/`, and the project state is marked `LIMITED` rather than claiming profiler coverage.
