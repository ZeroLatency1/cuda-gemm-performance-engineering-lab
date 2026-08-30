# Profiling Methodology

Profiling is intended to be performed using:
- **Nsight Systems (nsys):** For timeline analysis, kernel launch overhead, and general system behavior.
- **Nsight Compute (ncu):** For detailed SM utilization, memory throughput, occupancy, and register pressure.

## Pending Profiling

*Due to the lack of an NVIDIA GPU in the current environment, no profiler traces have been generated. Once run locally, this document will contain the hypotheses, metrics, observed behaviors, and conclusions for each major kernel.*
