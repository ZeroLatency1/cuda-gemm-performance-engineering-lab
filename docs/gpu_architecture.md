# GPU Architecture (Ada Lovelace / RTX 4060)

## Overview

The NVIDIA GeForce RTX 4060 Laptop GPU is based on the Ada Lovelace architecture (Compute Capability 8.9).

## Key Characteristics

- **Streaming Multiprocessors (SMs):** The fundamental unit of execution. Warps of 32 threads are scheduled on the SM.
- **Tensor Cores (4th Gen):** Accelerated matrix multiplications, capable of executing mixed-precision math (FP16, BF16, TF32).
- **Registers:** Large register file per SM. High register pressure per thread can limit occupancy.
- **Shared Memory / L1 Cache:** Configurable fast on-chip memory.
- **L2 Cache:** Larger cache shared across all SMs before hitting global memory.

## Execution Model
- **SIMT:** Single Instruction, Multiple Threads. 32 threads (a warp) execute the same instruction.
- **Occupancy:** The ratio of active warps to the maximum possible warps. Higher occupancy can help hide latency, though maximum occupancy is not always required for peak performance if latency is hidden via instruction-level parallelism or sufficient memory bandwidth.

*(Note: Detailed empirical observations on the RTX 4060 for this specific project are pending execution).*
