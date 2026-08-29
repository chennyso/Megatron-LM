# PP/VPP Trace Findings on g5 RTX 5090

All runs use the same 32-layer dense GPT proxy, bf16, sequence length 1024,
global batch 8, seed `20260829`, and `NCCL_P2P_DISABLE=1` unless noted.  The
machine has no usable GPU-GPU P2P (`nvidia-smi topo -p2p r/w` reports `CNS` for
every non-self pair), so PP activation traffic is host/PCIe mediated.

## Dominant path

Nsight reports show that interleaved PP is communication-bound on this
consumer topology:

| plan | SendRecv kernel share | cudaStreamSynchronize API share |
| --- | ---: | ---: |
| PP4/TP2/VPP2 | 72--73% | 37--39% |
| PP4/TP2/VPP4/8 | 71--73% | 38--39% |
| PP8/TP1/VPP4 | 82.8% | 54.4% |

The reports are under
`/workspace/runs/g5-8gpu-strategy-research-20260829/` on g5.

## Topology permutation falsification

PP4/TP2/VPP4 was run with three physical rank orders.  The non-Nsight run
medians over the last 15 steps (including occasional long-step outliers) were
approximately 296.2 ms (default `0,1,2,3,4,5,6,7`), 293.1 ms (NUMA grouped),
and 290.1 ms (deliberately cross-paired).  This apparent improvement did not
replicate under paired Nsight: default had median 360.9 ms and cross-pair
control 368.0 ms.  Nsight also measured SendRecv average 2.50 ms versus 2.67
ms and stream-sync share 38.9% versus 41.5%, respectively.  The single-run
step-time difference is therefore noise; cross-NUMA TP pairing is worse.

## Boundary-layout falsification

We tested Megatron's flexible pipeline layout strings for PP4/TP2/VPP2 and
PP8/TP1.  In three interleaved repeats (50 iterations each), PP4 aggregate
medians were 287.0 ms (uniform), 288.4 ms (boundary-light), and 300.7 ms
(first-heavy).  PP8 aggregate medians were 215.9 ms (uniform), 220.8 ms
(boundary-light), and 220.1 ms (first-heavy).  Moving one layer around the
embedding boundary does not improve this uniform dense workload and is not a
valid standalone contribution.

## Current actionable conclusion

The reproducible optimization opportunity is not a fixed VPP depth or a
hand-written uneven layout.  The planner must choose PP/TP/VPP jointly from
measured phase costs and the PCIe/NUMA topology: PP8/TP1 is substantially
faster than PP4/TP2 on this machine, while deeper VPP eventually saturates on
PP P2P and synchronization.  Any proposed runtime strategy must therefore
change the critical communication path or prove a topology-conditioned choice;
priority-only or boundary-only tweaks are falsified by these traces.
