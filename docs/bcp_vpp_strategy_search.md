<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# BCP-VPP Strategy Search

BCP-VPP is a bounded critical-path search loop for Megatron interleaved virtual
pipeline parallelism. The method keeps Megatron correctness checks in the
training path and moves strategy generation, ranking, and profiling evidence to
offline tools.

The core claim is narrow: PP/VPP performance should be optimized at the
virtual-chunk critical path, not only by equal layer counts, lower bubble ratio,
or more visible communication/compute overlap.

## Motivation

Existing automatic pipeline systems usually emphasize one of these axes:

- Static schedule expression and search, as in FlexPipe, Koala, Tessel, and
  GraphPipe style systems.
- Runtime readiness or asynchronous dispatch, as in RRFP and AMDP-like work.
- Load partitioning under imbalance or interference, as in LIBPipe and DynPipe.
- Communication overlap or transport primitives, as in overlap runtimes and
  DeepEP-style pipeline discussions.

BCP-VPP targets the gap between those axes in Megatron-LM:

- Megatron VPP has a concrete interleaved 1F1B schedule table and tensor queue
  invariants.
- The best candidate may be a different VP degree, microbatch group size,
  boundary layout, or placement, even when the default schedule table remains
  unchanged.
- More overlap is not automatically better. Useful overlap hides P2P on the
  critical path; harmful overlap slows compute kernels; fake overlap appears on
  the timeline but does not shorten the step.

## Search Space

BCP-VPP exposes a legality-preserving rewrite algebra:

| Rewrite | Purpose | Current backend status |
| --- | --- | --- |
| `baseline` | Preserve Megatron default interleaved VPP | Executable |
| `front_loaded_group` | Reduce initial group exposure | Verifier-protected; fixed backend rejects unsafe non-default tables |
| `change_group_size` | Tune `microbatch_group_size_per_vp_stage` | Executable through plan override |
| `change_vp_degree` | Tune virtual stages per PP rank | Executable when layer divisibility holds |
| `boundary_layout` | Move safe layer boundaries via pipeline layout | Executable through plan override |
| `node_local_placement` | Prefer local critical PP edges | Metadata for launch/topology integration |

The important implementation choice is that candidates go through
`StrategyVerifier`. If a schedule-table rewrite would violate current Megatron
queue lifetime assumptions, it is rejected instead of being silently benchmarked
as an invalid schedule.

## Objective

For each candidate, the searcher combines trace-derived and static DAG signals:

```text
score =
  predicted_critical_path
+ exposed_p2p_wait
+ chunk_skew_penalty
+ activation_budget_violation
+ forward_backward_delay_violation
+ p2p_credit_violation
```

The report records these concrete fields:

- `bcp_critical_path_ms`
- `bcp_static_task_dag_critical_path_ms`
- `bcp_exposed_p2p_wait_ms`
- `bcp_activation_peak_mb`
- `bcp_fb_delay_steps`
- `bcp_chunk_skew`
- `bcp_p2p_credit_pressure`
- `bcp_score`

This is intentionally bounded. It does not claim an aggressive asynchronous
runtime unless tagged P2P and tensor lifetime management are implemented.

## Algorithm

1. Run a short Megatron profiling job with `pipeline_strategy_trace_path`.
2. Attribute bottlenecks from rank, chunk, P2P wait, memory, and readiness
   signals.
3. Generate constrained rewrite proposals with a deterministic proposer or an
   optional LLM proposer.
4. Expand proposals into bounded candidates over group size, VP degree, layout,
   and placement.
5. Build a `StrategyTask` DAG for each candidate and compute the static longest
   path.
6. Score candidates with the BCP objective and budget violations.
7. Verify candidates through `StrategyVerifier`.
8. Emit a `StrategyPlan` plus a candidate report.
9. Validate the selected plan with a short training run and optional Nsight
   Systems export.

An LLM is not trusted as the optimizer. Its role is only to emit schema-valid
proposal hints. The deterministic searcher, verifier, and profiling loop decide
what can run and what improves measured performance.

## Code Paths

- `megatron/core/pipeline_parallel/schedules.py`
  consumes `pipeline_strategy_policy`, `pipeline_strategy_plan`, trace hooks,
  and the existing Megatron VPP schedule table.
- `megatron/core/pipeline_parallel/strategy_synthesizer.py`
  defines `StrategyPlan`, `StrategyTask`, rewrite records, and the verifier.
- `tools/pipeline_strategy_agent.py`
  turns traces into bottlenecks and rewrite proposals.
- `tools/search_pipeline_strategy.py`
  ranks candidates with BCP and writes the executable plan.
- `tools/analyze_bcp_vpp_trace.py`
  produces read-only BCP diagnostics from traces.
- `tools/analyze_effective_overlap.py`
  classifies useful, harmful, fake, and exposed communication overlap.
- `tools/run_bcp_vpp_loop.py`
  produces the complete offline experiment artifact bundle.

## Example Offline Loop

```bash
python tools/run_bcp_vpp_loop.py \
  --trace runs/trace_rank*.json \
  --output-dir runs/bcp_vpp_loop \
  --num-microbatches 16 \
  --num-model-chunks 4 \
  --microbatch-group-size 4 \
  --pipeline-parallel-size 4 \
  --num-layers 64 \
  --candidate-budget 24 \
  --objective bcp \
  --bcp-activation-budget-mb 28000 \
  --bcp-p2p-credit-budget 8 \
  --bcp-fb-delay-budget 64
```

The loop writes:

- `agent_proposals.json`
- `candidate_report.json`
- `best_strategy.json`
- `bcp_trace_analysis.json`
- `effective_overlap.json`
- `manifest.json`

Run the selected plan by passing:

```bash
--pipeline-strategy-plan runs/bcp_vpp_loop/best_strategy.json
```

## Nsight Validation

Export an Nsight Systems report to SQLite, then join it with Megatron traces:

```bash
nsys export --type sqlite --output run.sqlite run.nsys-rep

python tools/analyze_effective_overlap.py \
  --trace runs/trace_rank*.json \
  --nsys-sqlite run.sqlite \
  --output runs/effective_overlap.json
```

Use the output as a validation table:

- Useful overlap: communication is covered by compute and compute duration is
  not slower than its solo median.
- Harmful overlap: communication overlaps compute, but the compute kernel is
  slower than its solo median by the configured threshold.
- Fake overlap: communication appears in the same timeline span but is not
  covered by compute, or does not reduce exposed wait.

This distinction is important for consumer GPUs and PCIe-only clusters, where
forcing overlap can compete with GEMM or memory traffic.

## Experimental Protocol

Compare at least these variants:

- Megatron default VPP.
- Manual group-size tuning.
- BCP-selected group size.
- BCP-selected VP degree.
- BCP-selected boundary layout.
- BCP-selected combined plan.

Report:

- Median step time after warmup.
- Throughput tokens/s.
- Pipeline bubble or exposed wait from trace.
- Peak activation/runtime memory.
- `bcp_static_task_dag_critical_path_ms`.
- Useful/harmful/fake overlap from `analyze_effective_overlap.py`.
- Candidate rejection reasons from `candidate_report.json`.

The ablation should show whether improvement comes from group size, VP degree,
layout, placement, or overlap quality. If BCP does not improve a setting, the
rejection/report artifacts should still explain why the default was kept.

## Limitations

- Full out-of-order VPP schedule execution is not enabled in the fixed backend.
  Tagged P2P and explicit tensor lifetime ownership are required before that is
  safe.
- `fb_delay_steps` is an event-order proxy, not a formal staleness proof.
- Nsight SQLite schemas vary across versions; the overlap parser uses defensive
  column discovery and should be validated on the target Nsight release.
- The current node-local placement field is metadata until launcher topology
  binding consumes it.

These limits are deliberate. The current contribution is a Megatron-safe,
profile-guided search framework for PP/VPP partition and strategy selection,
not a speculative asynchronous runtime.
