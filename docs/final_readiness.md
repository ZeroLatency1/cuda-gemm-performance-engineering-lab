# Final Readiness Checklist

This fresh starter is the canonical merge of the original KUCH project surface and the new experiment-preservation infrastructure.

Before the project is called complete on the target machine:

- [ ] Clean build succeeds with CUDA 12.0.140 / SM 8.9.
- [ ] Correctness suite passes.
- [ ] Baseline measurements are captured.
- [ ] Every benchmark appends to the experiment history.
- [ ] `NOT_VERIFIED` is distinct from `FAIL`.
- [ ] CUDA-event timing is used for GPU kernel timing.
- [ ] GFLOPS uses the same latency statistic that is reported.
- [ ] Tensor Core correctness is verified for supported dimensions.
- [ ] cuBLAS baseline is measured with a persistent handle.
- [ ] Profiling is completed where the environment permits it, or the limitation is documented.
- [ ] Optimization experiments are recorded with their hypotheses and measured deltas.
- [ ] Final summaries and plots are generated only from stored measurements.
- [ ] Documentation matches observed implementation and measurements.
