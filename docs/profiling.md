# Profiling Methodology

## Nsight Compute

Use Nsight Compute to answer a specific kernel hypothesis. Examples include register pressure, memory throughput, occupancy, instruction issue, or whether Tensor Core instructions are present.

The supplied `scripts/run_profile.sh` captures a register-tiled 2048³ run with a timestamped export path. Replace or extend the command only when a new hypothesis justifies another experiment.

## Nsight Systems

Nsight Systems is optional in this target environment. It is appropriate for system-level timeline questions such as transfer overlap, launch overhead, and synchronization behavior.

## Failure handling

A profiler failure is not converted into a fake success. The profile script stores the timestamp, tool, exact command, exit code or missing-tool condition under `profiles/limitations/`. The project state is marked `LIMITED` rather than claiming profiler coverage.
