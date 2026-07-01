# 8x5090D Observation Study

This directory turns the paper-oriented 8x5090D observation study into a
reproducible workflow that can be:

- edited locally inside the Megatron-LM fork
- pushed to GitHub from the current branch
- pulled directly by a Volcano pod on `g5` / `g6`
- summarized into paper-ready figures and tables

## Scope

The workflow is organized around five artifact groups:

1. `Fig. 1 / Table 1`: hardware and communication profile
2. `Fig. 2 / Table 2-3`: baseline throughput, memory, feasibility
3. `Fig. 3`: utilization and per-rank behavior over iterations
4. `Fig. 4 / Table 4`: representative NSight Systems critical-path breakdown
5. `Fig. 5 / Table 5`: manual rewrite before/after validation

## Layout

```text
experiments/observation_8x5090d/
├── README.md
├── configs/
│   └── observation_matrix.json
├── figures/
│   ├── paper_plot_style.py
│   ├── gen_fig1_hardware_profile.py
│   ├── gen_fig2_baselines.py
│   ├── gen_fig3_utilization.py
│   ├── gen_fig4_nsys_breakdown.py
│   └── gen_fig5_rewrites.py
├── k8s/
│   └── volcano_single_node_8gpu.yaml.tmpl
├── runner/
│   └── observation_entrypoint.sh
└── scripts/
    ├── aggregate_observation_results.py
    ├── render_volcano_job.py
    ├── run_hardware_profile.py
    ├── run_megatron_observation.py
    └── summarize_megatron_run.py
```

## Important Notes

- The current Megatron tree in this workspace already contains Qwen3-8B,
  Qwen3-14B, and Qwen3-30B-A3B-compatible configs. The "fast dense sweep"
  therefore uses `Qwen3-8B` as the dense small-model proxy unless you replace
  its dimensions with an exact 7B variant.
- `--pipeline-model-parallel-layout` is the official path for non-uniform or
  asymmetric pipeline partitioning in this Megatron version.
- `--use-megatron-fsdp` can be combined with pipeline parallelism and MoE here;
  `--use-torch-fsdp2` cannot.

## Recommended Workflow

### 1. Edit locally

Work inside this repository:

```bash
cd /Users/chenny/Documents/k8s使用/Megatron-LM
```

### 2. Push your branch

The current fork already has a GitHub remote (`origin`). Push the active branch
after reviewing changes:

```bash
git status --short
git add experiments/observation_8x5090d
git commit -m "Add 8x5090D observation study scaffold"
git push origin "$(git rev-parse --abbrev-ref HEAD)"
```

### 3. Launch a single-node 8-GPU job

Render a Volcano manifest and optionally apply it:

```bash
python experiments/observation_8x5090d/scripts/render_volcano_job.py \
  --phase hardware \
  --node g5 \
  --job-name obs5090d-hw-g5 \
  --run-id obs5090d-hw-$(date +%Y%m%d-%H%M%S) \
  --git-branch "$(git rev-parse --abbrev-ref HEAD)" \
  --apply
```

Useful phases:

- `hardware`
- `baseline`
- `nsys`
- `rewrite`

For `baseline`, `nsys`, or `rewrite`, pass `--case-id <case_id>` to target a
single matrix entry.

### 4. Aggregate results

After jobs complete, aggregate raw logs:

```bash
python experiments/observation_8x5090d/scripts/aggregate_observation_results.py \
  --result-root /workspace/runs/observation_8x5090d/<run-id> \
  --output-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
```

### 5. Generate figures

```bash
python experiments/observation_8x5090d/figures/gen_fig1_hardware_profile.py \
  --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python experiments/observation_8x5090d/figures/gen_fig2_baselines.py \
  --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python experiments/observation_8x5090d/figures/gen_fig3_utilization.py \
  --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python experiments/observation_8x5090d/figures/gen_fig4_nsys_breakdown.py \
  --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
python experiments/observation_8x5090d/figures/gen_fig5_rewrites.py \
  --analysis-dir /workspace/runs/observation_8x5090d/<run-id>/analysis
```

## Output Contract

Each run stores:

- raw command outputs
- exact case configuration JSON
- Megatron logs
- `nvidia-smi dmon` samples
- parsed summary JSON per case
- aggregated CSV / JSON tables for plotting

That contract is what makes later paper claims auditable.
