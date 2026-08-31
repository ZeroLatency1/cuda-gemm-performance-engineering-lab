# Memory Hierarchy

1. **Global Memory:** Largest capacity, highest latency, lowest bandwidth. Crucial to ensure memory coalescing.
2. **L2 Cache:** Shared across all SMs.
3. **L1 / Shared Memory:** On-chip. Shared memory allows cooperative data loading to drastically reduce global memory bandwidth requirements in GEMM.
4. **Registers:** Fastest memory. Used in register tiling to keep data as close to the ALU as possible and minimize shared memory accesses.

*(Note: Detailed bandwidth utilization metrics are pending execution).*
