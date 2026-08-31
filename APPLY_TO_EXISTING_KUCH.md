KUCH CUDA GEMM — HARDENED OVERLAY

Apply this archive on top of the existing KUCH project. It intentionally excludes
results/experiment history, profiles, .git metadata, and project_state.json so
those live artifacts are not overwritten.

From the existing project directory:
  unzip -o /path/to/KUCH_CUDA_GEMM_HARDENED_OVERLAY.zip
  ./scripts/test_static.sh

After applying, run your target-side build/verification/benchmark commands as usual.
The new profiling script auto-discovers Nsight Compute and links each successful
profile to the benchmark-generated EXP-XXXXXX under profiles/<EXP-ID>/.
