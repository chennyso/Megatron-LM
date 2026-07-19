# Strategy Results: Endpoint-Conditioned P2P Policy

## Workload

- 2 nodes: g5 + g6, 16 x RTX 5090, SR-IOV IB
- Qwen3-32B, 64 layers, hidden size 5120, BF16
- TP2 / PP8, sequence length 4096, global batch 16
- FineWeb-Edu formal slice, no FSDP
- 10 warmup steps + 10 measured steps per repeat
- Measurement: median and mean of measured iteration times from rank 15

## Throughput Confirmation

The strongest existing Megatron baseline is VPP8 + SP + overlapped P2P. Its five
independent medians are 8398.90, 8407.35, 8408.80, 8398.45, and 8413.45 ms.

The candidate policy is:

```text
TP2 / PP8 / VPP8
sequence parallel = on
P2P overlap = off
warmup-flush overlap = off
microbatch group size = 8
uniform layer cuts
```

Its five independent medians are 8357.65, 8369.45, 8365.60, 8356.10, and
8355.85 ms.

| configuration | mean ms/iter | tokens/s | within/between CV |
|---|---:|---:|---:|
| VPP8 + SP + overlap (n=5) | 8405.39 | 7802.5 | 0.078% baseline between-run |
| VPP8 + SP + no-overlap (n=5) | 8360.93 | 7844.0 | 0.074% |

The improvement is 44.46 ms/iter, or 0.529%. Welch comparison over the five
independent run medians gives an approximate 95% CI of [35.16, 53.76] ms for
the improvement (df = 7.98), so the interval excludes zero.

## Mechanism Evidence

Matched Nsight Systems captures use the same 10-step warmup and 2-step capture
window on the same Qwen3-32B workload.

| kernel | overlap | no-overlap |
|---|---:|---:|
| `ncclDevKernel_SendRecv` instances | 8064 | 3008 |
| `ncclDevKernel_SendRecv` total time | 23.22 s | 19.02 s |
| dominant BF16 GEMM total time | 26.85 s | 26.81 s |

The no-overlap policy removes 62.7% of SendRecv kernel instances while leaving
the dominant GEMM time effectively unchanged. This supports an
endpoint/action-conditioned contention explanation: on this dual-node
consumer-GPU topology, overlapped P2P launches add NCCL work on the same
critical path as GEMMs instead of hiding it.

## Negative Ablations

- `VPP4` nonuniform edge-balanced layout `[1,3,2x28,3,1]`: 8658.68 ms/iter,
  2.83% slower than uniform VPP4/no-SP.
- `VPP8` nonuniform edge-balanced layout `[0,2,1x60,2,0]`: 8685.19 ms/iter,
  3.33% slower than the strongest VPP8/SP baseline.
- `VPP8 + SP + warmup/flush overlap`: 29092.1 ms/iter, 71.11% slower.
- Generated `seam-staggered` schedule with first group size 4 at PP8 was
  rejected after it triggered the existing executor's input-queue
  `IndexError`; the verifier now rejects any group smaller than PP.

The current executable method is therefore a measured policy gate, not an
arbitrary schedule rewrite: retain P2P overlap only when the profiler predicts
that endpoint contention is below the measured threshold; otherwise lower to
the no-overlap action policy.
