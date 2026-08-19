# TP-lane 假设的排除结果

## 实验

- 配置：`TP=2, PP=8, VPP=3, DP=1`，48 层，hidden 4096，sequence 2048，global batch 16。
- 拓扑：两节点各 8 张 RTX 5090，PP stage 3 到 4 跨节点。
- 运行：24 个训练 iteration；Nsight 只在 global rank 6 的 step 8--13 捕获，训练在 capture 停止后完整结束。
- 正确性：PP payload 配对、P2P request lifecycle、collective tickets、DP replica bucket shape 全部通过。

## 观察

在 trace 中用 `(source_pp, tp, vp, microbatch, virtual_microbatch, direction)` 做 metadata-only 配对，共配对 28,066 条消息。两个 TP lane 的 host-observed request wait 几乎相同：

| lane | sender mean | receiver mean | receiver P95 |
|---|---:|---:|---:|
| TP0 | 0.00660 ms | 0.00584 ms | 0.00934 ms |
| TP1 | 0.00598 ms | 0.00490 ms | 0.00759 ms |

Nsight rank 6 的 CUDA API 摘要中，`cudaEventSynchronize` 占 API 时间 88.1%；GPU kernel 摘要中 `ncclDevKernel_SendRecv` 占 44.4%。这说明 PP 路径存在显著的同步/通信工作，但不能归因于 TP0 的线级服务不对称。

## 结论边界

原先由聚合 `p2p_wait` 得到的 TP0/TP1 差异被精确消息配对否定。它可能来自 producer readiness、schedule API 包含的 host 时间或早期 trace 的上下文缺失。不能把“TP-lane asymmetry”写成研究发现，也不能据此设计 lane-aware partition。

下一步应针对 `cudaEventSynchronize` 的调用归属、VPP action 的发送/接收 completion deadline 和完整 step tail 做因果控制，而不是继续优化 TP lane 映射。
