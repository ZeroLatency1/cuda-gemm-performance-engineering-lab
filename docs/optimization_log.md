# Optimization Log

This file is intentionally evidence-driven. A row is only added after the implementation is built and measured on the target environment.

| Experiment ID | Hypothesis | Change | Baseline | Median latency | GFLOPS | Delta | Verification | Decision |
|---|---|---|---|---:|---:|---|---|---|
| PENDING | No target-side measurement has been executed in the current environment. | — | — | — | — | — | — | PENDING |

## Recording rule

For each meaningful optimization, record the experiment ID, hypothesis, change, parent/baseline IDs, measured median/p95/minimum latency, GFLOPS, correctness result, and accept/reject decision. Performance claims must be traceable to `results/experiments.csv` / `.jsonl` records.
