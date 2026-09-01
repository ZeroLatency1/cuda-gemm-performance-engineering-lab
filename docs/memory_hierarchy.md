# Memory Hierarchy

1. **Global Memory:** Largest capacity, highest latency, lowest bandwidth. Crucial to ensure memory coalescing.
2. **L2 Cache:** Shared across all SMs.
3. **L1 / Shared Memory:** On-chip. Shared memory allows cooperative data loading to drastically reduce global memory bandwidth requirements in GEMM.
4. **Registers:** Fastest memory. Used in register tiling to keep data as close to the ALU as possible and minimize shared memory accesses.


## Measured profile observations

The completed Nsight Compute profiles show high on-chip activity for several kernels while reported external DRAM throughput remains comparatively low in the sampled profiles. This supports analyzing register reuse, cache behavior, and execution resources rather than assuming every GEMM variant is DRAM-bandwidth-bound. The values are profile-specific and should not be generalized beyond the measured workloads.
