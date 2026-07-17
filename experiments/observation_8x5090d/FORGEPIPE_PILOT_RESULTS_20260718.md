# ForgePipe Pilot Results and Historical Feasibility Anchors

Date: 2026-07-18

## Evidence Status

This document separates engineering validation, historical feasibility evidence, and
paper-grade measurements. R000/R000N and all recovered historical runs are not main-paper
performance results. R000 uses a 0.6B structural sanity model. R000N includes Nsight
observer overhead. The recovered 32B and 30B-A3B runs use mock data and NullTokenizer,
so they establish system and strategy feasibility but not data-pipeline or convergence
validity.

## Current Pilot Runs

| Run | Configuration | Result | Interpretation |
|---|---|---|---|
| R000 | 7xRTX 5090, 0.6B structure, PP7/VPP2, group 7, seq 2048, GBS14, BF16, no FSDP | 6/6 steps; steady mean 656.47 ms and 43,678 tokens/s; rank peak allocated 3.23-8.01 GiB | Trace and runtime sanity only |
| R000N | Same topology with Nsight 2026.2.1, profile ranks 0 and 6 | 8/8 steps; return code 0; complete `.nsys-rep`, SQLite, three stats CSV files, and seven strategy traces | Profiling pipeline sanity; profiled step times are excluded from throughput claims |

R000N produced 21,541 CUDA kernel records, 73,947 NVTX events, and 100,287 CUDA
runtime records. The seven strategy traces contain 977 events: 196 forward, 196 backward,
293 P2P issue, and 292 P2P wait events. Every trace event has non-null host start and end
timestamps.

Aggregated kernel residency across all seven GPUs is:

| Category | Kernel time | Share | Instances |
|---|---:|---:|---:|
| NCCL | 2,485.04 ms | 69.54% | 406 |
| Compute (CUTLASS/GEMM/FlashAttention/softmax) | 819.74 ms | 22.94% | 6,706 |
| Other | 268.76 ms | 7.52% | 14,429 |

`ncclDevKernel_SendRecv` alone accounts for 63.1% of summarized kernel residency.
This is evidence for investigating P2P placement and overlap, not evidence that 63.1% of
wall time is exposed communication. Kernel residency overlaps across devices and streams;
the paper claim must use critical-path timeline analysis.

The R000N summary currently includes profiled iterations in its steady-state aggregate:
step 6 is 657.3 ms, while steps 7 and 8 rise to 2,623.7 and 8,302.8 ms during capture and
report finalization. Therefore its reported 3,861.27 ms mean is an observer-effect artifact.

## Recovered Dual-Node Evidence

The old `chenny-workspace` PVC contains auditable logs showing that model size alone does
not explain previous failures.

### Qwen3-32B structure

Common configuration: 2 nodes, 16xRTX 5090, TP2/PP8, seq 4096, microbatch 1, GBS16,
BF16, full uniform recompute, distributed optimizer, no Megatron FSDP. Means use steps
3-8 from one completed eight-step run.

| Schedule | Mean step time | Speedup over 1F1B | Completion |
|---|---:|---:|---:|
| 1F1B | 9,754.97 ms | baseline | 8/8 |
| VPP2 | 8,606.73 ms | 13.34% | 8/8 |
| VPP4 | 8,074.77 ms | 20.81% | 8/8 |
| VPP8 | 7,967.08 ms | 22.44% | 8/8 |

VPP8 is only 1.35% faster than VPP4. A publishable method must compare against VPP8,
not only against 1F1B. The same historical workspace also contains failed 32B layouts
and schedule variants, confirming that the feasible set is strategy-dependent.

### Qwen3-30B-A3B structure

The recovered command instantiates 48 layers, hidden size 2048, 128 experts, top-8
routing, MoE FFN size 768, seq 4096, BF16, grouped GEMM, full recompute, and the
distributed optimizer.

| Topology | GBS | Mean step time | Completion |
|---|---:|---:|---:|
| TP2/PP8/VPP2/EP1 | 16 | 4,845.77 ms (steps 3-12) | 12/12 |
| TP2/PP4/VPP2/EP2 | 64 | 19,860.37 ms (steps 3-8) | 8/8 |

These two rows are not a matched performance comparison because GBS and EP differ.
They prove that the full MoE compute/optimizer/communication graph can run on the two
consumer-GPU nodes. Formal paper runs must replace mock data with the pinned FineWeb-Edu
slice and record correctness checks.

### Existing tuning and Agent baseline

The historical SlackVPP sweep found that changing `CUDA_DEVICE_MAX_CONNECTIONS` from 1
to 16 improved its strongest 30B-A3B configuration by only 0.82%. This is an important
negative result: a single overlap-capacity knob is not enough for the new paper.

The historical AgentPipe run was heuristic, not LLM-driven. It diagnosed backward/forward
skew, 19.97% traced P2P wait, memory pressure, and chunk/rank imbalance, then proposed
group, boundary, checkpoint, and split-wgrad changes. Its validated plan nevertheless
remained default placement, global group 8, and fixed runtime. The measured steady state
was essentially unchanged. ForgePipe must close this proposal-to-executable-policy gap.

## Failure Taxonomy

The following are engineering/setup failures and must not count as strategy improvements:

- nonexistent checkpoint or tokenizer paths;
- invalid Megatron arguments or illegal global group values;
- missing CLI lowering for a strategy field;
- warning text misclassified as a runtime failure;
- profiler rank encoding errors;
- obsolete Nsight report names or missing profiler artifacts.

A failure enters the strategy dataset only if the configuration is valid, all ranks start
the intended model, the data/math/precision contract matches the control, and the terminal
cause is OOM, schedule/order deadlock, communication timeout, or statistically worse
steady-state throughput.

## Refined Optimization Target

The recovered experiments expose two coupled constraints:

1. Uniform VPP already captures most of the easy bubble reduction; VPP8 leaves little
   headroom over VPP4.
2. Naive non-uniform partitions that assign seven MoE layers to one physical rank OOM,
   even when their compute balance looks better.

ForgePipe should therefore synthesize a compensated boundary move rather than a layer
move alone. One candidate program jointly specifies:

- contiguous layer-to-physical-rank and layer-to-virtual-chunk assignment;
- per-chunk recompute window and microbatch run length;
- guarded forward/backward priority at ambiguous ready points;
- P2P post and wait placement without changing the NCCL signature.

The verifier enforces exactly-once execution, dependency order, matching communication,
and a hard per-rank memory bound. The objective minimizes calibrated critical-path time:

`T_hat = T_compute_cp + T_exposed_p2p + T_bubble + T_contention_penalty`

subject to `M_peak(rank) <= M_budget(rank)` for every rank. Agent code is only retained as
a paper contribution if, under equal candidate and GPU-minute budgets, it beats the
strongest non-LLM synthesizer by at least 3% on two workloads. The systems claim requires
at least 5% over fully tuned Megatron with a 95% confidence interval excluding zero.

## Next Confirmatory Runs

1. Reproduce the 32B 1F1B/VPP2/VPP4/VPP8 anchor with pinned commit and five independent
   runs; use VPP8 as the tuned baseline.
2. On 8B single-node, measure uniform VPP and memory-feasible compensated boundary moves
   with identical kernels, tokens, and recompute work.
3. Add timeline overlap analysis so accumulated NCCL residency is converted into exposed
   P2P wait and critical-path attribution.
4. Compare layout-only, schedule-only, and joint programs before enabling LLM generation.

Formal R001-R006 remain blocked until eight standard physical GPU resource slots are
available on one node. Seven-GPU runs are engineering sanity only.
