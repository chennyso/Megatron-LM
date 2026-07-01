# 8x5090D Observation Study

This directory implements the single-node `8x5090D` observation study for a
paper-grade systems evaluation. The workflow is designed around one rule:
every claim must be reproducible from raw logs, repeated-run summaries, and
figure scripts.

## Scope

- Main paper scope: single-node `g5` with `8 x RTX 5090`
- Main storage: `NFS PVC` (`seampipe-paper-workspace`)
- Main model store: `chenny-models-nfs`
- Main public workload: deterministic `FineWeb-Edu` slice on NFS
- Local corpus: sanity-only, never enters the main paper figures/tables
- Proxy workloads: smoke/shakeout only, excluded from main paper claims

## Artifact Contract

Each repeat produces:

- raw `stdout.log` / `stderr.log` / `combined.log`
- `nvidia-smi dmon` samples
- `command.sh`
- `run_metadata.json`
- `case_config.json`
- `summary.json`
- `per_step.csv`
- `rank_memory.csv`
- `gpu_dmon_samples.csv`

Each aggregated run directory produces:

- `run_summaries.csv`
- `case_aggregates.csv`
- `table2_feasibility_matrix.csv`
- `table5_rewrite_pairs.csv`
- `figure_data/*.csv`
- copied hardware artifacts

## Important Note About The Small Dense Model

The current Megatron tree in this workspace has a runnable `Qwen3-8B` path,
plus `Qwen3-14B` and `Qwen3-30B-A3B`. The matrix therefore uses:

- `Qwen3-8B` for proxy shakeout
- `Qwen3-14B` and `Qwen3-30B-A3B` for the formal matrix already wired here

If you want an exact `Qwen3-7B` formal section, add the exact 7B checkpoint and
verified hyperparameter block first, then extend `observation_matrix.json`
rather than silently reusing the 8B config.

## Workflow

### 1. Prepare the public dataset slice

```bash
cd /Users/chenny/Documents/k8s使用/Megatron-LM
python3 experiments/observation_8x5090d/scripts/prepare_observation_dataset.py \
  --matrix-path experiments/observation_8x5090d/configs/observation_matrix.json \
  --dataset-spec-id formal_public_fineweb_edu_v1 \
  --output-root /workspace/datasets/fineweb_edu/formal_slice \
  --overwrite
```

Proxy slice:

```bash
python3 experiments/observation_8x5090d/scripts/prepare_observation_dataset.py \
  --matrix-path experiments/observation_8x5090d/configs/observation_matrix.json \
  --dataset-spec-id proxy_public_subset_v1 \
  --output-root /workspace/datasets/fineweb_edu/proxy_slice \
  --overwrite
```

### 2. Launch hardware profiling

```bash
python3 experiments/observation_8x5090d/scripts/render_volcano_job.py \
  --phase hardware \
  --node g5 \
  --job-name obs5090d-hw-g5 \
  --run-id obs5090d-hw-$(date +%Y%m%d-%H%M%S) \
  --git-branch "$(git rev-parse --abbrev-ref HEAD)" \
  --apply
```

### 3. Launch proxy or formal cases

Proxy smoke:

```bash
python3 experiments/observation_8x5090d/scripts/render_volcano_job.py \
  --phase proxy \
  --node g5 \
  --job-name obs5090d-proxy-g5 \
  --run-id obs5090d-proxy-$(date +%Y%m%d-%H%M%S) \
  --case-id proxy_qwen8b_pp2_fsdp \
  --git-branch "$(git rev-parse --abbrev-ref HEAD)" \
  --apply
```

Formal baseline:

```bash
python3 experiments/observation_8x5090d/scripts/render_volcano_job.py \
  --phase baseline \
  --node g5 \
  --job-name obs5090d-baseline-g5 \
  --run-id obs5090d-baseline-$(date +%Y%m%d-%H%M%S) \
  --case-id qwen14b_pp4_fsdp \
  --git-branch "$(git rev-parse --abbrev-ref HEAD)" \
  --apply
```

NSYS representative:

```bash
python3 experiments/observation_8x5090d/scripts/render_volcano_job.py \
  --phase nsys \
  --node g5 \
  --job-name obs5090d-nsys-g5 \
  --run-id obs5090d-nsys-$(date +%Y%m%d-%H%M%S) \
  --case-id qwen14b_nsys_baseline \
  --git-branch "$(git rev-parse --abbrev-ref HEAD)" \
  --apply
```

### 4. Aggregate outputs

```bash
python3 experiments/observation_8x5090d/scripts/aggregate_observation_results.py \
  --result-root /workspace/runs/observation_8x5090d/<run-id> \
  --output-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
```

### 5. Generate figures

```bash
python3 experiments/observation_8x5090d/figures/gen_fig1_hardware_profile.py --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python3 experiments/observation_8x5090d/figures/gen_fig2_baselines.py --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python3 experiments/observation_8x5090d/figures/gen_fig3_utilization.py --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python3 experiments/observation_8x5090d/figures/gen_fig4_nsys_breakdown.py --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python3 experiments/observation_8x5090d/figures/gen_fig5_rewrites.py --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
```

Each figure script writes:

- `PDF`
- `SVG`
- `600 DPI PNG`
- `600 DPI TIFF`

## Schema Highlights

Each case in `configs/observation_matrix.json` now records:

- `paper_model_id`
- `dataset_spec`
- `repeat_policy`
- `warmup_steps`
- `measure_steps`
- `profiler_policy`
- `figure_membership`
- `claim_membership`

That schema is the bridge between raw runs and paper claims.
