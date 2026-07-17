# FoldDuplex-VPP: Dual-Node Research Design

## 1. Paper Decision

The first paper should not attempt to optimize FSDP, TP, CP, EP, offload, arbitrary GraphPP actions, and kernels at once. The primary setting is resident-weight, synchronous Megatron-LM training on two 8 x RTX 5090 nodes connected by IB:

- fixed outer parallelism: TP2 + PP8, with each TP group kept inside one node;
- optimized variables: nonuniform logical-stage cuts, virtual-stage-to-physical-rank route, and boundary P2P issue/pair/wait policy;
- main model: Qwen3-32B, sequence 4096, real pretraining data;
- optional external validity: Qwen3-14B and Qwen3-30B-A3B after the dense path passes;
- no FSDP in the first paper;
- no asynchronous or stale-weight semantics.

Working method name: **FoldDuplex-VPP**. Working system name: **ForgePipe-Agent**.

The central claim is not that an LLM can tune `PP=8,VPP=4`. It is that an Agent can generate a verified executable route/schedule policy that changes the action graph, while a deterministic optimizer instantiates layer boundaries and communication pairings from measured topology costs.

## 2. Evidence Already Established

### 2.1 Historical model anchor

The recovered 32B TP2/PP8 runs used IB on both nodes. Each schedule log contains 16 `NET/IB : Using` records per node and selects `IBext_v11`; no Socket transport selection appears.

| Megatron schedule | Mean step time, iterations 3-8 |
|---|---:|
| 1F1B | 9,754.97 ms |
| VPP2 | 8,606.73 ms |
| VPP4 | 8,074.77 ms |
| VPP8 | 7,967.08 ms |

VPP8 is 22.44% faster than 1F1B but only 1.35% faster than VPP4. Therefore the correct baseline is tuned VPP8, and VPP4 is a promising route/cut operating point if its repeated cross-node communication can be reduced.

The historical 32B and 30B-A3B runs use mock data/NullTokenizer. They prove model-structure and memory feasibility only; formal claims require real data, pinned model/tokenizer paths, repetitions, and current commits.

### 2.2 Current interconnect anchor

The 2026-07-18 link profile proves:

- 24.57 GB/s one-way P2P at 128 MiB;
- 47.38 GB/s aggregate bidirectional P2P;
- NCCL RDMA Plugin v11 with GPUDirect RDMA;
- two complete Nsight Systems reports.

Thus the research problem is not repairing IB. It is avoiding unnecessary VPP crossings and exploiting full duplex on the crossings that remain.

## 3. Literature-to-Gap Matrix

The local evidence base contains 37 full technical readings. The table lists the works that directly constrain this paper.

| Work | What it actually optimizes | What it closes | Remaining gap used here |
|---|---|---|---|
| PithTrain, 2026 preprint | compact Agent-native framework, task skills, DualPipeV/compile implementation | Agent efficiency and compact modification surface | does not synthesize PP cuts, VPP routes, or schedule code |
| GraphTrainer Autoresearch, 2026 project | Agent edits graph passes, benchmarks, bitwise gate, retain/reject | generic Agent generate-measure loop is not novel | distributed route/action synthesis under topology constraints |
| GraphPP, 2026 technical talk/RFC | AOTAutograd F/B capture, dI/dW split, FSDP collective extraction, graph multiplex | executable schedule IR and graph transforms | does not choose stage cuts, VPP route, or action order automatically |
| RoundPipe, 2026 preprint | offloaded stateless workers, F/B-asymmetric cuts, round-robin dispatch | consumer-GPU offload PP and additive `t_max` partitioning | resident-weight synchronous VPP and shared-link contention |
| Operator-MIP, 2026 preprint | MIP discovery of symbolic F/dI/dW schedules | generic solver-based schedule discovery | no real LLM lowering, measured IB contention, or route verifier |
| RRFP, 2026 preprint | readiness-driven dynamic execution under runtime variability | ready-set fallback and nonbinding hints | planned topology route and duplex pairing for stable dense training |
| Tabular Schedule, ISPDC 2026 | table-to-DAG evaluation with communication/memory costs | bubble ratio alone is insufficient | hardware-calibrated route synthesis and executable lowering |
| HARP, 2025/2026 | fine-layer nonuniform cuts and H-1F1B for heterogeneous clusters | nonuniform partition by itself is not novel | homogeneous consumer GPUs with VPP crossing multiplicity/full duplex |
| HetAuto, EuroSys 2026 | MCTS over PP/DP/CP/TP and layer distribution across slow clusters | MCTS tuple/cut search is covered | it fixes contiguous cluster stages and does not alter VPP route/action code |
| Tangram, 2025/2026 | compose homogeneous planner outputs across topology islands | PP composition of existing partial plans | generation of a route primitive absent from the underlying planner |
| DIP, ASPLOS 2026 | multimodal segmenting, MCTS order, online F/B queues, memory ILP | dynamic multimodal scheduling is covered | fixed dense workload, boundary-specific full-duplex matching |
| Galvatron | per-layer hybrid parallel strategy search | Agent selecting per-layer DP/TP/PP/FSDP overlaps prior art | virtual-stage route and executable P2P schedule below tuple level |
| ZeroPP | ZeRO-3 + VPP scheduling units, dI/dW delay | parameter reuse and VPP/FSDP interaction | excluded by the no-FSDP scope |
| Seq1F1B, NAACL 2025 | sequence slicing and sequence-level PP | nonuniform sequence partition is covered | model-stage route and link-direction pairing |
| TawPipe, AAAI 2026 | topology-aware weight-passing pipeline | topology awareness and weight movement | activation-passing resident-weight Megatron with unchanged math |

Peer-reviewed venue and preprint status must remain separate in the paper. PithTrain, RoundPipe, RRFP, and Operator-MIP are current preprints; GraphPP evidence is a technical talk/RFC, not a full archival paper.

## 4. Core Technical Observation

Let `P` be physical PP degree and `V` virtual chunks per physical rank. Standard interleaved Megatron maps each chunk in the same physical direction:

```text
chunk 0: 0 -> 1 -> ... -> P-1
chunk 1: 0 -> 1 -> ... -> P-1
...
```

With two contiguous node islands, each chunk crosses the node boundary once and each chunk transition wraps from `P-1` back to `0`, crossing again. The number of forward cross-node logical edges is therefore:

```text
C_standard(V) = V + (V - 1) = 2V - 1.
```

Folded routing alternates physical direction:

```text
chunk 0: 0 -> 1 -> ... -> P-1
chunk 1: P-1 -> P-2 -> ... -> 0
chunk 2: 0 -> 1 -> ... -> P-1
...
```

The end of one chunk and start of the next occupy the same endpoint, eliminating wrap crossings:

```text
C_folded(V) = V.
```

At VPP8 this removes 7 of 15 cross-node logical edges in both forward and backward, a 46.7% structural reduction. This is not yet a step-time claim because communication may overlap with compute.

The second observation is measured rather than assumed: opposite-direction P2P reaches 47.38 GB/s aggregate versus 24.57 GB/s one-way. Remaining forward and backward boundary actions should therefore be paired when their legal execution windows overlap.

## 5. FoldDuplex-VPP Algorithm

### 5.1 Profiled action graph

For every candidate logical stage `s` and microbatch shape, measure:

```text
tF(s), tB(s), activation_bytes(s -> s+1), peak_live_bytes(s)
tIB_oneway(bytes), tIB_duplex(bytes_f, bytes_b)
```

Build a DAG containing F, B, activation send/receive, parameter/gradient update deadlines, and buffer lifetimes. Compute earliest release and latest legal start for each cross-node communication action.

### 5.2 Joint route and nonuniform cut

For a route family `rho(c,k) -> physical_pp_rank`, choose contiguous layer boundaries `b_0...b_{PV}` and minimize the calibrated critical path:

```text
min  CP(DAG(rho, b, q))

subject to
  every layer appears exactly once and in model order,
  every physical rank owns V logical stages,
  TP groups remain intra-node,
  peak_live_bytes(rank) <= 32 GiB - safety_headroom,
  forward and backward routes are exact reverses,
  every P2P send has one matching receive.
```

`q` is the P2P issue/pair/wait policy. The inner cut optimizer is dynamic programming over contiguous layers for a fixed route and bottleneck bound. A small CP-SAT/MIP instance acts as an oracle on reduced models; Operator-MIP-style symbolic scheduling is a baseline, not the contribution.

### 5.3 Full-duplex boundary matching

Let `F_i` be g5-to-g6 boundary actions and `B_j` be g6-to-g5 actions. A pair is legal when their `[release, deadline]` windows intersect and simultaneous buffers fit. Its measured saving is:

```text
w_ij = tIB_oneway(F_i) + tIB_oneway(B_j)
       - tIB_duplex(F_i, B_j).
```

Select nonconflicting pairs that maximize total saving. The initial implementation uses maximum-weight bipartite matching followed by list scheduling on the event DAG. The verifier replays the final action list and rejects unmatched collectives, cyclic dependencies, buffer reuse before completion, or memory overflow.

### 5.4 Agent-generated policy code

The Agent does not sit in the training hot path and does not estimate exact integer optima. It receives:

- the compact route/schedule API and tests;
- layer and link profiles;
- counterexamples such as repeated IB crossings, exposed waits, OOM rank, or failed pairing;
- a fixed candidate and GPU-minute budget.

It writes a typed `TopologySchedulePolicy` implementation with:

```text
match/guard -> route macro -> cut constraints -> pairing rule
-> proof obligations -> Megatron lowering
```

The deterministic inner optimizer instantiates boundaries and microbatch ids. Static verification, differential correctness, timeout/OOM isolation, and real hardware measurement retain or reject the patch. This follows PithTrain's small explicit modification surface, but unlike PithTrain the Agent changes an executable parallel strategy.

## 6. Why Performance Can Improve

The expected gain has three separable sources:

1. **Route effect**: fewer cross-node activation/gradient edges.
2. **Duplex effect**: remaining opposite-direction edges use both IB directions concurrently.
3. **Cut effect**: nonuniform stages reduce the compute bottleneck and create legal communication slack without exceeding per-rank memory.

Historical VPP4 is only 107.69 ms slower per step than VPP8. Folded VPP4 reduces default cross-node edge count from 7 to 4 while retaining fewer/larger virtual stages and a more flexible nonuniform cut space. It is therefore the first high-value candidate, followed by folded VPP8.

This reasoning establishes a deterministic traffic reduction, not guaranteed end-to-end speedup. The paper proceeds only if matched real-model runs show at least 5% over the best fully tuned Megatron baseline with a confidence interval excluding zero. If route-only gains are below this threshold, duplex pairing and cut balance must account for the remainder; otherwise the main claim is rejected.

## 7. Strong Baselines

Only three baseline families belong in the main paper:

1. **Fully tuned Megatron**: TP2/PP8, VPP2/4/8, flexible layout, legal microbatch grouping, P2P overlap settings, and rank placement. VPP8 is the historical anchor.
2. **Non-Agent synthesis**: DP/CP-SAT/evolutionary search over the same fixed route and pairing primitives, with an Operator-MIP-style oracle on small instances.
3. **Agent controls**: Agent-knobs-only and Agent-code under exactly the same candidate count, profiler inputs, verifier, and GPU-minute budget.

GraphPP, RoundPipe, RRFP, HARP, HetAuto, and Galvatron are mechanism/related-work comparisons. They should be run end-to-end only when their execution semantics and target workload can be matched fairly; otherwise the paper must not manufacture an apples-to-oranges speedup table.

## 8. Claim and Kill Gates

| Claim | Required evidence | Kill condition |
|---|---|---|
| C1: FoldDuplex-VPP improves resident-weight dual-node training | >=5% step-time improvement over tuned VPP8 on primary 32B setting, 5 repeats, paired 95% CI above zero; Nsight shows fewer cross-node actions and/or more duplex overlap | improvement <3% after correct tuning, or gain disappears without profiler instrumentation |
| C2: Agent code generation is useful | >=3% better best-found throughput than strongest equal-budget non-LLM synthesis on two settings, or materially lower time-to-best with equal final quality; at least one reusable valid macro outside fixed grammar | deterministic search matches Agent within 3%; remove Agent from title and primary claim |
| Correctness | same token count and optimizer semantics; loss/grad tolerance fixed before throughput tests; no unmatched P2P or deadlock | any speedup changes work, precision, batch, or update semantics |

## 9. Experiment Matrix

### Main paper

- Qwen3-32B, TP2/PP8, VPP4 and VPP8, seq 4096, real FineWeb-Edu, BF16.
- Tuned Megatron, folded-route only, duplex only, nonuniform-cut only, joint FoldDuplex, Agent-generated policy.
- 20 warmup + 50 measured iterations, 5 independent runs, randomized configuration order.
- Throughput runs without Nsight; representative matched runs with Nsight on boundary ranks.
- Report step time, tokens/s, MFU, peak memory, exposed P2P wait, cross-node action count/bytes, duplex overlap, critical-path idle, search GPU-minutes, and valid-patch rate.

### Appendix/external validity

- Qwen3-14B dense.
- Qwen3-30B-A3B only after dense correctness and routing are stable.
- Sequence 2048 and 8192 sensitivity.
- IB-disabled diagnostic solely to validate the cost model, not as a weak baseline.
- Small-instance exhaustive/MIP oracle regret.

### Intentionally excluded

- FSDP/ZeRO residency optimization.
- stale-gradient asynchronous schedules.
- lossy activation compression.
- arbitrary CUDA kernel generation.
- single-node 0.6B results as paper performance evidence.

## 10. Immediate Engineering Order

1. Add a pure route verifier and crossing-count unit tests for standard and folded VPP.
2. Add a route-aware logical predecessor/successor API to `P2PCommunicator`; keep default behavior byte-for-byte compatible.
3. Lower folded VPP2 on a two-node small model and verify exact forward/backward/loss equivalence.
4. Add boundary-action timestamps and measure whether standard VPP already pairs opposite directions.
5. Implement weighted duplex matching only if measured unpaired legal windows exist.
6. Run 32B folded VPP4/VPP8 only after 8 standard physical GPU slots are schedulable on both g5 and g6.

Current scheduling caveat: GPUs are physically idle, but g5's Mirage system components reserve five `nvidia.com/gpu` resources and g6 has a `stage0-excluded` taint. The diagnostic used an explicit taint toleration. Formal 8+8 runs remain blocked until standard physical GPU accounting exposes all eight slots per node; no paper claim should use a mixed Mirage/standard allocation without a separately approved protocol.

