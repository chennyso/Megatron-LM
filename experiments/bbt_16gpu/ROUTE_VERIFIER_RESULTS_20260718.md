# FoldDuplex Route Verifier Result

**Run ID**: R101  
**Date**: 2026-07-18  
**Code commit**: `591876406` plus the subsequent isort-only test-file fix  
**Scope**: Static route semantics only; no training throughput claim

## Environment

- Node: `g5` (CPU-only Pod; no GPU resource requested)
- Image: `harbor.bbt.sspu.edu.cn/nvcr/nvidia/pytorch:26.05-py3`
- Python: 3.12.3
- Test dependency: pytest 9.1.1 installed in the ephemeral Pod
- Source files:
  - `megatron/core/pipeline_parallel/route_policy.py`
  - `tests/unit_tests/pipeline_parallel/test_route_policy.py`

## Verified Properties

`PipelineRoute` represents logical model order as a total ordering of
`(virtual_chunk, physical_rank)` pairs. The verifier checks:

1. every pair occurs exactly once;
2. backward dependencies exactly reverse forward dependencies;
3. local virtual-chunk transitions are not lowered to NCCL P2P;
4. every remote edge lowers to exactly one endpoint-correct send and receive;
5. missing receives and wrong peers are rejected.

For two contiguous four-rank node islands and PP8:

| VPP chunks | Standard crossings | Folded crossings | Reduction |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.0% |
| 2 | 3 | 2 | 33.3% |
| 4 | 7 | 4 | 42.9% |
| 8 | 15 | 8 | 46.7% |

The formulas exercised by the parameterized tests are:

```text
C_standard(V) = 2V - 1
C_folded(V) = V
```

## Test Output

```text
...............                                                          [100%]
15 passed in 0.02s
```

The same container also passed the repository's `isort --check-only` check for
both files.

## Interpretation and Remaining Gate

R101 establishes that the proposed logical route and its communication
signatures are internally consistent. It does not prove that Megatron can yet
execute the route. Folded execution additionally requires route-aware stage
endpoints, local chunk-to-chunk tensor handoff, dynamic remote neighbors, and
warmup/cooldown formulas. R102 remains blocked on that lowering and must pass
same-seed loss/gradient equivalence and a deadlock timeout before any performance
run.
