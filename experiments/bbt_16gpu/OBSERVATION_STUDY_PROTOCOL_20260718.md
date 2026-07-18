# Dual-Node 16-GPU Observation Study Protocol

## Scope

This protocol freezes the measurement process, not the proposed optimization.
The primary workload is Qwen3-32B pretraining on real FineWeb-Edu tokens with
TP2/PP8, no FSDP, BF16, and resident stage weights on `g5 + g6` (16 RTX 5090
GPUs over SR-IOV InfiniBand). A method is selected only after the baseline
measurements identify an exposed bottleneck.

Single-GPU, 1+1-GPU, mock-data, and NullTokenizer runs are engineering checks.
They cannot enter the main result table or establish a performance claim.

## Research Questions

1. How much of the VPP2/VPP4/VPP8 gain over 1F1B comes from bubble reduction?
2. Where does VPP stop scaling, and is the plateau caused by small compute
   chunks, launch overhead, stage imbalance, memory pressure, or exposed P2P?
3. Does standard VPP expose repeated cross-node wrap traffic, or is that
   traffic already hidden by compute?
4. Are opposite-direction IB transfers serialized when legal full-duplex
   overlap windows exist?
5. Which measured bottleneck has enough recoverable critical-path time to
   support a paper-level speedup over a fully tuned Megatron baseline?

## Frozen Workload

| Item | Value |
|---|---|
| Model | Qwen3-32B architecture, 64 layers, hidden 5120, FFN 25600 |
| Data | deterministic 65,536-document FineWeb-Edu indexed slice |
| Parallelism | TP2/PP8/DP1, no FSDP, no CP, no EP |
| Sequence and batch | sequence 4096, MBS 1, GBS 16 |
| Precision | BF16 with the same precision-aware optimizer in every case |
| Recomputation | full, uniform, one layer |
| Measurement | iterations 1-20 warmup; iterations 21-70 measured |
| Repetition | five independent process launches per throughput case |
| Ordering | seeded random order across case and repeat |
| Throughput profiling | disabled |
| Attribution profiling | separate trace and Nsight runs |

Changing tokens, precision, optimizer semantics, recomputation, batch size, or
model dimensions invalidates a paired comparison.

## Execution Blocks

### O1: Schedule screening

Run 1F1B, VPP2, VPP4, and VPP8 with Megatron's legal default communication
behavior. VPP uses microbatch group size 8. Each case receives five randomized
throughput repeats.

This block answers whether the historical VPP4/VPP8 plateau reproduces with
real tokens and a current Megatron commit. Historical mock-data numbers are
anchors only and are not pooled with these results.

### O2: Communication controls

Run VPP2/VPP4/VPP8 with P2P overlap disabled. For the two best schedules, test
group size 16 and warmup/flush P2P overlap. These are baseline tuning controls,
not proposed-method ablations.

The strongest legal configuration, rather than a fixed VPP8 label, becomes the
paper baseline.

### O3: Trace attribution

Run one separate trace job for each screened schedule. Only TP rank 0 and DP
rank 0 emit each PP-stage trace. The trace records:

- forward and backward duration by PP rank, virtual chunk, and microbatch;
- P2P issue and blocking-wait duration;
- send/receive direction metadata;
- allocated-memory snapshots;
- warmup, steady, and cooldown phase for non-interleaved 1F1B.

CUDA-event timing synchronizes the measured operation and therefore has an
observer effect. Trace durations explain bottlenecks but never replace the
unprofiled throughput result.

### O4: Nsight attribution

Run one separate Nsight Systems job per screened schedule. Capture two steady
iterations after warmup with `cuda,nvtx,osrt`; retain one report per node plus
SQLite and CSV exports. NCCL INFO logs must show the RDMA plugin and no Socket
fallback.

Nsight analysis reports kernel launch count, GEMM duration distribution, NCCL
P2P calls, cross-node boundary traffic, exposed CPU/CUDA wait, overlap, and the
fraction of step time in warmup/steady/cooldown ranges.

## Metrics And Definitions

| Metric | Definition |
|---|---|
| Step time | median of iterations 21-70 within a run; run median is the statistical sample |
| Throughput | `GBS * sequence_length / step_seconds` |
| Between-run variation | CV over five run medians |
| Stage compute imbalance | `(max stage F+B - min stage F+B) / mean stage F+B` |
| Chunk imbalance | F and B duration summarized by `(pp_rank, virtual_chunk)` |
| Exposed P2P wait | blocking P2P wait on the rank critical path, not total NCCL kernel time |
| Boundary actions | P2P issue events at the PP3/PP4 physical node boundary |
| Boundary bytes | Nsight/NCCL payload bytes for the same boundary |
| Bubble/idle | critical-span time not covered by F, B, or non-exposed communication |
| Duplex overlap | intersection of opposite-direction boundary-transfer intervals divided by their union |
| Peak memory | maximum allocated and reserved memory per physical rank |
| Correctness | identical consumed tokens and optimizer semantics; finite loss/grad norm with matched trend |

For throughput claims, use the five run medians and a paired bootstrap 95%
confidence interval. Require between-run CV at or below 3%; otherwise add
repeats and diagnose environmental noise before comparing methods.

## Evidence-To-Method Decision

| Observation | Primary optimization candidate | Reject when |
|---|---|---|
| Cross-node wraps contribute exposed wait and scale with VPP | route/placement synthesis that reduces physical crossings | crossing time is hidden and step time does not track it |
| PP-rank or chunk compute has material skew | profile-driven nonuniform contiguous cuts with memory constraints | skew is below 3% or embedding/loss placement already removes it |
| Opposite directions have legal but serialized windows | duplex-aware boundary action matching | standard Megatron already overlaps the windows |
| VPP8 adds many short GEMMs/launches and barely beats VPP4 | adaptive VPP coarsening or heterogeneous virtual chunk sizes | plateau is instead communication- or memory-bound |
| Blocking P2P waits occur after late issue | release-aware P2P issue/wait scheduling | waits are dominated by unavoidable dependency latency |
| Peak memory, not time, is the limiting factor | revise the no-FSDP thesis before method development | memory headroom is sufficient and uncorrelated with stalls |

The first implemented optimization must explain at least 5% predicted
recoverable critical-path time. A structural count reduction alone is not
enough.

## Agent Role After Observation

The Agent is introduced only after a deterministic optimization has a measured
performance mechanism. It receives the frozen workload, topology, trace
summary, legal strategy IR, and verifier API. It may generate or modify a small
parallel-policy module that chooses cuts, virtual chunks, placement, or P2P
action order. It may not change model math, tokens, precision, optimizer,
kernel implementations, or the measurement harness.

Every generated patch is statically verified, differential-tested, and run in
an isolated candidate job. Equal-budget comparisons include DP/CP-SAT,
evolution/beam search, Agent-knobs-only, and Agent-code. Agent code is a paper
contribution only if it finds a valid strategy at least 3% better than the
strongest equal-budget non-LLM method on two settings, or reaches equal quality
with materially fewer real-hardware trials.

## Current Execution State

### Invalidated screening evidence

The run family `obs16-r103-screening-final-20260718` is retained only as
contaminated engineering evidence.  Its interleaved path constructed a CUDA
event timer and synchronized the end event for every forward and backward
virtual microbatch even when strategy tracing was disabled.  The number of
observer-induced synchronizations therefore increased with VPP granularity,
while the 1F1B path bypassed the same instrumentation.  Those results must not
be used for schedule ranking, speedup, or paper claims.  All screening cases
must be repeated from an immutable commit after timer gating, with a tuned
sequence-parallel baseline included.

The same invalid run family also used `--kv-channels 80`, while the mounted
Qwen3-32B `config.json` and Megatron's Qwen3-32B reference configuration use a
128-dimensional attention head.  Formal runs use `--kv-channels 128`; the
earlier workload must not be described as the Qwen3-32B architecture.

The real dataset, tokenizer, model config, IB link characterization, renderer,
randomized matrix controller, 1F1B/VPP trace paths, and analysis scripts are
ready. A two-node real-data shakeout was submitted as
`obs16-obs16-r103-shakeout-20260718-screen-1f1b-r1`.

It is currently gang-pending because Kubernetes accounting reports fewer than
eight standard `nvidia.com/gpu` slots on both nodes: the g5 Mirage plugin and
topology collector request five slots in total, and the g6 topology collector
requests one. These are infrastructure workloads with no visible compute, but
they cannot be deleted, evicted, or bypassed by the experiment harness. The
queued job will become runnable when the standard 8+8 allocation is restored.
