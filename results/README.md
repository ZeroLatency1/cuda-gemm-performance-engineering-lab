# Experiment History

`experiments.csv` and `experiments.jsonl` are the official append-only histories. Each attempted configuration receives a unique `EXP-XXXXXX` ID. `results/raw/` stores an immutable JSON copy for each experiment.

Statuses:

- `PASS`: execution completed; `verification_status` separately says whether numerical verification passed.
- `UNSUPPORTED`: configuration is not supported by that kernel path and is explicitly recorded.
- `ERROR`: execution or validation failed; performance numbers must not be interpreted as successful evidence.

`verification_status=NOT_VERIFIED` means a performance-only run used `--no-verify`. It is not a correctness failure.
