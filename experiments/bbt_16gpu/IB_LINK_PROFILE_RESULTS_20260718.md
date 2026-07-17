# g5-g6 IB/NCCL Link Profile

**Run ID**: `forgepipe-ib-link-20260718`  
**Commit**: `d371fbb8ecec665695c8d9dcb4f39683c4780892`  
**Local date**: 2026-07-18  
**Status**: PASS as an interconnect characterization; not a model-throughput result.

## Configuration

| Item | Value |
|---|---|
| Nodes | g5 + g6 |
| GPUs | 1 x RTX 5090 per node, 32,607 MiB |
| Network | one `sriov-ib-network` VF per Pod |
| HCA | g5 `mlx5_34`, g6 `mlx5_20` |
| Link | 200 Gb/s IB, active MTU 4096 |
| Runtime | PyTorch 2.12.0a0 nv26.05, CUDA 13.2, NCCL 2.30.4 |
| Transport proof | `NET/IB : Using`, `NCCL RDMA Plugin v11`, GPUDirect RDMA DMABUF |
| Payloads | 1, 4, 16, 32, 64, 128 MiB BF16 |
| Repeats | 5 per payload and operation |
| Operations | one-way P2P, simultaneous bidirectional P2P, two-rank all-reduce |

NCCL printed warnings for unassigned host HCAs before successfully opening the one SR-IOV VF visible to each Pod. These warnings do not indicate Socket fallback. No `NET/Socket` selection appears in either log.

## Measured Results

`GB/s` is decimal. Bidirectional P2P reports aggregate traffic in both directions. The per-message time is the mean measured loop duration divided by iteration count; initialization and warmup are outside the CUDA-event interval.

| Payload | P2P one-way GB/s | One-way ms/message | P2P bidirectional aggregate GB/s | All-reduce GB/s |
|---:|---:|---:|---:|---:|
| 1 MiB | 16.24 | 0.065 | 28.56 | 13.33 |
| 4 MiB | 22.55 | 0.186 | 40.37 | 18.18 |
| 16 MiB | 24.32 | 0.690 | 45.81 | 19.10 |
| 32 MiB | 24.42 | 1.374 | 46.48 | 20.09 |
| 64 MiB | 24.52 | 2.737 | 47.33 | 20.80 |
| 128 MiB | 24.57 | 5.463 | 47.38 | 21.16 |

At 128 MiB, one-way P2P reaches 98.3% of the nominal 25 GB/s unidirectional line rate. Simultaneous opposite-direction P2P reaches 94.8% of the nominal 50 GB/s full-duplex aggregate. Therefore a schedule that pairs forward and backward boundary transfers can use a hardware capability that a serialized schedule leaves idle.

## Nsight Evidence

Both nodes produced a valid `.nsys-rep` and SQLite export. The dedicated link benchmark contains no model compute, so kernel proportions characterize the benchmark itself rather than a training bottleneck:

| GPU kernel family | g5 total time share | g6 total time share | Instances per node |
|---|---:|---:|---:|
| `ncclDevKernel_SendRecv` | 62.4% | 62.5% | 6,880 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 36.6% | 36.7% | 3,440 |

NVTX ranges identify every operation, payload, and repeat. The first one-way 1 MiB range includes communicator initialization in host-range duration, but the reported bandwidth excludes warmup through CUDA-event timing.

## Implication for PP/VPP Optimization

The link is not broken and is not the primary novelty. The actionable facts are:

1. Cross-node activation transfers of 16-64 MiB cost about 0.69-2.74 ms each even at line rate.
2. Default VPP can repeat the same physical node crossing for every virtual chunk and at chunk wraparound.
3. Opposite-direction transfers can nearly double aggregate link utilization when issued together.
4. A useful optimizer must minimize the number of cross-node logical edges and deliberately align remaining forward/backward transfers, rather than merely selecting a larger VPP degree.

These measurements parameterize the topology term in the FoldDuplex-VPP event model. End-to-end speedup remains an experimental hypothesis until a matched 32B training run beats tuned Megatron VPP8.

## Artifacts

- Local raw logs and reports: `/Users/chenny/Documents/Codex/2026-07-13/w/results/forgepipe-ib-link-20260718/`
- Persistent cluster copy: `/workspace/results/forgepipe-ib-link-20260718/` on `seampipe-paper-workspace`
- Benchmark: `experiments/bbt_16gpu/scripts/benchmark_ib_link.py`
- Volcano manifest: `experiments/bbt_16gpu/k8s/volcano_ib_link_2node.yaml`

