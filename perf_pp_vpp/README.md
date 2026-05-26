# perf_pp_vpp

`perf_pp_vpp` is a controller-driven profiling pipeline for reproducible PP/VPP experiments on the
2 nodes x 8 GPUs RTX 5090D + InfiniBand Megatron-LM environment.

## Goals

- Compare `PP=8,DP=2` vs `PP=16,DP=1`
- Sweep `VPP=1/2/4` and measure bubble, message count, and exposed P2P time
- Inspect the cross-node `stage7 -> stage8` boundary
- Test overlap switches: `overlap_p2p_comm`, `overlap_grad_reduce`, `overlap_param_gather`
- Produce structured CSV / JSON / Markdown / figure outputs suitable for research reporting

## Execution model

These scripts are designed to run from a control machine that has:

- this repository checked out locally
- `kubectl` access to the two long-lived worker pods
- the same Megatron-LM code available on the shared PVC at `REMOTE_PROJECT_ROOT`

In the current cluster setup, the source of truth is the shared repo mounted in the worker pods at:

- `/workspace/code/Megatron-LM`

The controller scripts write outputs back into that shared path under:

- `/workspace/code/Megatron-LM/perf_pp_vpp/outputs`

The worker pods themselves do not need `kubectl`.

## Git-based pod sync

For repeatable remote execution, treat GitHub as the source of truth and update the worker pods to a
specific branch or commit before running experiments:

```bash
cd /path/to/Megatron-LM
git push origin <branch>
perf_pp_vpp/scripts/15_sync_remote_repo.sh --ref <branch-or-commit>
```

If the pod repo has local changes and you explicitly want to discard them:

```bash
perf_pp_vpp/scripts/15_sync_remote_repo.sh --ref <branch-or-commit> --force-clean
```

The sync script refuses dirty pod worktrees by default so you do not silently lose remote edits.
If the container trust store is incomplete and GitHub HTTPS verification fails, use:

```bash
perf_pp_vpp/scripts/15_sync_remote_repo.sh --ref <branch-or-commit> --no-ssl-verify
```

## Environment configuration

Edit [configs/env.sh](./configs/env.sh) and set:

- `NODE0_POD`, `NODE1_POD`, `K8S_NAMESPACE`
- `REMOTE_PROJECT_ROOT`
- `MODEL_SOURCE_PATH_QWEN3_14B`, `MODEL_SOURCE_PATH_QWEN3_32B`
- `TOKENIZER_PATH`
- optionally `NCCL_TESTS_ROOT`, `NSYS_BIN`, `PYTHON_BIN`, `TORCHRUN_BIN`

Default cluster values already point to:

- `REMOTE_PROJECT_ROOT=/workspace/code/Megatron-LM`
- `MODEL_SOURCE_PATH_QWEN3_14B=/workspace/models/qwen3-14B`
- `MODEL_SOURCE_PATH_QWEN3_32B=/workspace/models/qwen3-32B`
- `TOKENIZER_PATH=/workspace/models/qwen3-14B`

All variables can be overridden through the shell environment.

## Phase 1

```bash
cd /path/to/Megatron-LM
source perf_pp_vpp/configs/env.sh
perf_pp_vpp/scripts/run_phase1_qwen14b.sh
```

Phase 1 prioritizes Qwen3-14B as the pipeline development / validation model and runs the minimum set first.

## Phase 2

```bash
cd /path/to/Megatron-LM
source perf_pp_vpp/configs/env.sh
perf_pp_vpp/scripts/run_phase2_qwen32b.sh
```

Phase 2 focuses on Qwen3-32B as the main VPP / cross-node communication study model.

## Run a single experiment

Example:

```bash
cd /path/to/Megatron-LM
source perf_pp_vpp/configs/env.sh
perf_pp_vpp/scripts/02_prepare_mock_data.sh
perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
  --experiment-yaml perf_pp_vpp/configs/experiments/phase1_qwen14b.yaml \
  --experiment-name q14_pp8_dp2_tp1_vpp1 \
  --seq-len 4096 \
  --repeat-id 1 \
  --profile-mode none \
  --profile-ranks none \
  --overlap-mode baseline
```

## Open Nsight Systems

A generated `.nsys-rep` can be opened in GUI with:

```bash
nsys-ui perf_pp_vpp/outputs/runs/<experiment>/seq4096/repeat1/nsys/<file>.nsys-rep
```

## Report interpretation

Main report:

- [outputs/reports/perf_report.md](./outputs/reports/perf_report.md)

Key tables:

- [outputs/summary/all_experiments.csv](./outputs/summary/all_experiments.csv)
- [outputs/summary/by_experiment_mean_std.csv](./outputs/summary/by_experiment_mean_std.csv)
- [outputs/summary/skipped_experiments.csv](./outputs/summary/skipped_experiments.csv)

Figures:

- `outputs/figures/*.png`

## Known limitations

- Phase 1 uses mock / synthetic data by design, so loader realism is intentionally suppressed
- Nsight SQLite schemas vary between versions; unavailable fields are emitted as `null`
- `B_IN` and `B_W` are not always cleanly separable without deeper schedule/autograd instrumentation
- Consumer 5090D + PCIe + IB results should not be directly generalized to NVLink/H100 systems

## Add a new model or strategy

1. Add a model YAML under `configs/models/`
2. Point `MODEL_SOURCE_PATH_*` to a valid `config.json`
3. Add an experiment YAML under `configs/experiments/`
4. Reuse `03_run_megatron_experiment.sh` and `09_generate_report.py`
