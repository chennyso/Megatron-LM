# Two-Node Folded Route Autograd Pilot

**Run ID**: `forgepipe-r102a-r2-20260718`  
**Milestone**: R102a  
**Commit**: `9235e82e0595b7ba0ab2689a0c9aeafd0aa9a1e8`  
**Claim scope**: lowering correctness and communication-count mechanism only

## Setup

- Nodes: g5 + g6, one RTX 5090 per node
- Network: one `sriov-ib-network` VF per Pod
- HCAs: g5 `mlx5_6`, g6 `mlx5_9`; both reached `PORT_ACTIVE`
- Runtime: PyTorch `2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2, NCCL 2.30.4
- Profiler: Nsight Systems 2026.2.1 on both ranks
- Workload: four-stage MLP, batch 8, four microbatches, hidden size 256, FP32
- Correctness reference: same weights/data and microbatch accumulation order on each GPU

The standard physical route is `0 -> 1 -> 0 -> 1`; the folded route is
`0 -> 1 -> 1 -> 0`. Every stage input is detached at a logical boundary so
forward activation and backward activation-gradient handoff are both exercised.

## Correctness and Mechanism Result

| Route | Remote edges / direction / microbatch | Forward sends | Backward sends | Local transitions | Prediction max abs | Gradient max abs | Loss abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 3 | 12 | 12 | 0 | 0 | 0 | 4.47e-8 |
| Folded | 2 | 8 | 8 | 4 | 0 | 0 | 4.47e-8 |

Folded PP2/VPP2 therefore replaces one remote chunk-wrap edge per direction and
microbatch with a local handoff, reducing executed remote sends by 33.3% while
preserving predictions, loss, and every owned parameter gradient within the
pre-registered `2e-5` tolerance.

NCCL logs on both ranks report:

```text
Using network NCCL RDMA Plugin v11
NET/IB : GPU Direct RDMA (DMABUF) enabled
```

There is no `NET/Socket` selection in either log. Nsight independently reports
20 `ncclSend` and 20 `ncclRecv` host actions per rank across the two routes,
matching `(12 + 8)` sends and receives. Both traces contain 40
`ncclDevKernel_SendRecv` GPU kernel instances.

## Profiling Interpretation

The raw NVTX summaries show `validate_standard` around 114 ms and
`validate_folded` around 6.1 ms. This is not a valid speedup measurement. The
standard route runs first and its first P2P call creates a lazy two-rank NCCL
communicator; the corresponding `route_0_forward` range is about 61-62 ms and
dominates the difference. The tiny MLP, Nsight instrumentation, synchronous
`dist.send/recv`, and unrandomized route order also make timing unsuitable for a
paper throughput claim.

The defensible evidence from this run is limited to:

1. both route directions execute without deadlock over real IB;
2. local chunk handoff works in forward and backward;
3. folded routing executes the predicted lower number of remote messages;
4. numerical results and parameter gradients match the reference.

## Attempt Audit

The first attempt, `forgepipe-r102a-20260718`, exited before distributed
initialization because the launcher did not add the repository root to
`PYTHONPATH`. Both ranks failed at the same import. Commit `9235e82e0` fixed the
launcher and renamed the second run to keep failure and success artifacts
separate. The first attempt is an environment/launch failure, not a failed route.

## Artifacts

Local:

```text
/Users/chenny/Documents/Codex/2026-07-13/w/results/forgepipe-r102a-r2-20260718/
```

Cluster PVC:

```text
/workspace/results/forgepipe-r102a-r2-20260718/
```

Each location contains two launcher logs, two `.nsys-rep` files, and per-node
`nvtx_sum`, `cuda_gpu_kern_sum`, and `cuda_api_sum` CSV files. The local trace
sizes exactly match the PVC files.

## Remaining R102b Gate

This pilot uses a dedicated four-stage executor, not Megatron's interleaved
training loop. R102b must integrate route-aware endpoints and local handoff into
Megatron, then pass a Qwen small-model same-seed loss/gradient comparison for at
least 20 steps. Only after R102b may the project start the 32B FoldDuplex
throughput matrix.
