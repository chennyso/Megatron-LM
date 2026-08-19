# 16-GPU Parallel-Training Reproduction Campaign

This note records only measured results. It deliberately does not assign a
new system name or claim a new method before a causal intervention has been
validated.

## Environment

- Two nodes, `g5` and `g6`, eight RTX 5090D GPUs each.
- Megatron snapshot used by the remote runner: `/workspace/runs/phaseweaver-20260818/code`.
- `NCCL_P2P_DISABLE=1`, IB enabled through `net1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Dense proxy: 48 layers, hidden 4096, sequence 2048, global batch 16,
  `TP=2, PP=8, DP=1`.
- MoE proxy: 48 layers, hidden 2048, 32 experts, expert parallel 2,
  `TP=2, PP=4, DP=2`, all-to-all dispatcher.

## Reproduced controls

### VPP P2P overlap

For `TP2/PP8/VPP3/DP1`, all performance numbers below are from runs with
strategy tracing disabled. Iterations 3--22 are used after warmup.

| mode | repeat 1 mean (ms) | repeat 2 mean (ms) | median range (ms) |
|---|---:|---:|---:|
| Megatron VPP overlap | 1514.1 | 1515.0 | 1512.5--1514.1 |
| `--no-overlap-p2p-communication` | 1552.8 | 1549.8 | 1548.0--1550.6 |

The default VPP overlap is therefore a real baseline improvement of about
2.3--2.6% for this dense proxy. An earlier traced run incorrectly suggested
that no-overlap was faster; disabling trace output reversed the result. The
earlier observation was measurement perturbation and is excluded.

### DP gradient overlap

For `TP2/PP4/VPP3/DP2`, no-trace runs gave:

| mode | steady mean (ms) |
|---|---:|
| default | 1466.8 |
| four DDP buckets | 1433.4 |
| eight DDP buckets | 1434.5 |

This reproduces the expected value of fine-grained gradient overlap. It is a
baseline control, not a new contribution. Changing the VPP group size from 4
to 8 while using four buckets worsened the mean to 1447.0 ms.

### MoE expert communication overlap

The original 128-expert configuration OOMs during first optimizer-state
materialization, with only about 8 MiB free on a 32 GiB GPU. A 32-expert
all-to-all proxy runs successfully.

Two unprofiled repeats show a robust distributional tradeoff:

| mode | repeat 1 median / mean (ms) | repeat 2 median / mean (ms) |
|---|---:|---:|
| all-to-all baseline | 2155 / 2674 | 2194 / 2768 |
| EP communication overlap | 2745 / 2865 | 2619 / 2938 |

The overlap mode improves neither median nor mean in this configuration,
although its P95 is not consistently worse. This is a reproduction of a
known failure mode of fixed overlap policies, not evidence that Tessera has
been improved upon.

### VPP-group sensitivity of MoE EP overlap

The default group for `PP=4` is four microbatches per virtual stage. Raising
the legal VPP group to eight changes the action wave without changing model
ownership. Two additional no-trace repeats show:

| VPP group | A2A baseline median / mean (ms) | EP overlap median / mean (ms) |
|---|---:|---:|
| 8, repeat 1 | 2162 / 2722 | 3048 / 3084 |
| 8, repeat 2 | 2185 / 2743 | 3057 / 3040 |

This is a stable worsening, rather than a useful winner inversion. It shows
that the penalty of fixed EP overlap depends on the VPP action wave, but it
does not establish a new mechanism beyond existing joint-overlap work.

### Periodicity check

A 64-iteration untraced all-to-all baseline was used to test the stricter
claim that the VPP period creates persistent phase locking. After removing
the first two iterations, the 62-step sequence has a lag-4 autocorrelation
of `-0.226`. Means grouped by `iteration mod 4` are 2553, 2453, 2528, and
2487 ms. This is not evidence of a stable VPP-period resonance. The current
tail is therefore treated as routing/system variability, not as a new
periodic phase-control mechanism.

## Nsight mechanism evidence

Nsight capture was controlled by Megatron's `cudaProfilerStart/Stop`, with
only global rank 6 captured for steps 4--8.

| metric | all-to-all baseline | EP overlap |
|---|---:|---:|
| `cudaEventSynchronize` share of CUDA API time | 84.0% | 80.4% |
| `ncclDevKernel_SendRecv` share of GPU kernel time | 71.6% | 65.4% |
| `cudaStreamSynchronize` share | 2.4% | 6.0% |

The overlap policy removes some SendRecv kernel exposure but introduces more
stream synchronization and changes the collective mix. This is the concrete
mechanism behind the median/tail tradeoff. It is not equivalent to wire
latency: `cudaEventSynchronize` includes host-side completion and dependency
waiting.

## Negative results and scope boundary

- TP-lane message pairing found no meaningful TP0/TP1 service asymmetry.
- The first apparent dense no-overlap win disappeared when trace writing was
  disabled.
- Generic communication contention, MoE route variation, and fixed overlap
  tradeoffs are already covered by Piper, Tessera, Lagom, and related work.
- No new PP/VPP/CP/EP algorithm is claimed in this campaign yet.

## Next causal test

The remaining defensible direction is a **selective, precompiled overlap
policy** that keeps the existing SPMD ordering and changes only which VPP
communication edges are asynchronous under a measured deadline. It must be
tested against default overlap, no-overlap, and Megatron DDP bucket overlap on
the same all-to-all proxy, with no continuous trace writing. If it does not
beat the best fixed policy on complete-step median and P95 across repeats, it
will be discarded rather than named as a contribution.
