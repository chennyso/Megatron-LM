# Ordered-Seam VPP Layout: 16-GPU Evidence

This document records the current hardware evidence for a concrete PP/VPP
optimization. It is a pilot result, not a claim that the work is already a
finished OSDI/SOSP paper.

## Finding

Megatron's interleaved schedule flattens virtual stages in a fixed order. A
virtual stage boundary is therefore not equivalent to another boundary even
when the number of decoder layers assigned to each physical rank is the same.
The warmup, steady-state, and cooldown wavefront repeatedly encounter those
boundaries in order. Embedding and loss/LM-head stages are also not ordinary
decoder layers.

The optimization target is consequently an ordered layout, not only a vector
of per-rank layer totals:

\[
\min_{x_1,\ldots,x_{PV}}
  T_{\mathrm{grouped\text{-}1F1B}}(x_1,\ldots,x_{PV})
  + \lambda\sum_i \lvert c_{i+1}-c_i\rvert,
\]

where `x_i` is the decoder count at virtual stage `i`, `c_i` includes the
special embedding/loss cost, and `T` is evaluated on the complete grouped
1F1B dependency schedule. The seam term is only a search regularizer; the
acceptance criterion is the measured end-to-end run.

## Hardware setup

- 2 nodes, 8 RTX 5090 per node (16 GPUs total)
- Qwen3-8B geometry proxy, 36 decoder layers
- BF16, sequence length 4096, microbatch 1, global batch 16
- Megatron distributed optimizer, TP/PP/DP as shown below
- `overlap-grad-reduce`, four DDP buckets
- 12 iterations; steady statistic uses iterations 3--11

## End-to-end measurements

| topology | layout | steady mean (ms) | repeats | change |
|---|---|---:|---:|---:|
| TP2/PP4/VPP3/DP2 | uniform 3 layers per virtual stage | 2097.1 | 2 | baseline |
| TP2/PP4/VPP3/DP2 | `Ett|tttt|ttt|ttt|ttt|ttt|ttt|ttt|ttt|ttt|tttt|ttL` | 2005.4 | 2 | **-4.37%** |
| TP4/PP2/VPP2/DP2 | uniform 9 layers per virtual stage | 3006.6 | 2 | baseline |
| TP4/PP2/VPP2/DP2 | `Etttttttt|tttttttttt|tttttttttt|ttttttttL` | 2950.3 | 2 | **-1.87%** |

The first layout has decoder counts
`[2,4,3,3,3,3,3,3,3,3,4,2]`. Its physical-rank totals are
`[8,10,10,8]`, so it intentionally does not optimize only the maximum rank
layer count.

## Seam ablation

The same ordered layout family can be slower when the extra layers are placed
at different virtual seams. In the earlier PP4/VPP3 ablation, a seam-alternate
layout with a different placement of the extra layers reached 2107.2 ms,
slower than the uniform baseline, despite having comparable total work. This
is the evidence that the mechanism is virtual-stage order plus special-stage
placement, not “put fewer layers on the first and last rank”.

## Trace evidence

The semantic schedule trace reports the following aggregate `p2p_comm_wait`
times (ms) for PP4/VPP3:

| PP rank | uniform | ordered-seam |
|---:|---:|---:|
| 0 | 106.7 | 172.0 |
| 4 | 147.7 | 100.7 |
| 8 | **223.4** | **157.9** |
| 12 | 92.6 | 136.0 |

The total wait is not reduced uniformly. The layout redistributes when waits
occur relative to the grouped wavefront, removing the rank-8 hotspot that
otherwise determines the exposed makespan. PP2/VPP2 shows the same direction:
rank-8 wait falls from 81.5 ms to 65.2 ms.

## Search and verification

`tools/vpp_layout_search.py` enumerates bounded contiguous decoder-count
vectors, ranks candidates by a cheap lower-cost key, and runs the complete
grouped-1F1B simulator only on the selected profile budget. The emitted layout
still goes through Megatron's layout parser and the real training executor.
The unit checks in
`tests/unit_tests/pipeline_parallel/test_vpp_layout_search.py` cover depth,
special-stage ordering, and cycle-free schedule simulation.

## Scope

This is materially different from changing VPP degree or enabling a generic
communication overlap flag. The decision variable is the *global ordered
virtual-stage layout*, and the objective is the execution wavefront makespan.
The next required evidence is a third repeat, a second model geometry, and a
full paper baseline matrix against uniform VPP, rank-load-only balancing, and
profile-guided contiguous PP cuts.
