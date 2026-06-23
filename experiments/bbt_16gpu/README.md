# BBT Dual-Node 16x5090 SeamPipe Experiments

This directory contains the reproducible launcher assets for the dual-node
`g5 + g6` SeamPipe experiments on BBT Kubernetes.

## Workflow

1. Push the current Megatron branch to GitHub.
2. Launch a short profile run on BBT with trace collection enabled.
3. Run the offline search loop on the collected traces to emit a verified
   `StrategyPlan`.
4. Launch the full 16-GPU run with `--pipeline-strategy-plan`.

## Assumptions

- Namespace: `default`
- Nodes: `g5` and `g6`
- Workspace PVC: `chenny-workspace` mounted at `/workspace`
- Model PVC: `chenny-models-nfs` mounted at `/models`
- Repo clone path inside pods: `/workspace/repos/Megatron-LM`
- Git remote: `https://github.com/chennyso/Megatron-LM.git`

## Files

- `run_profile_and_search.sh`
  - Generates the profile command and offline search command.
- `volcano-megatron-seampipe-16gpu.yaml`
  - Two-node Volcano job template for profile or full SeamPipe runs.

## Recommended models

- Fast profile / loop validation: `Qwen/Qwen3-0.6B`
- Main 16-GPU paper experiments: `Qwen/Qwen3-8B`

## Typical loop

```bash
bash experiments/bbt_16gpu/run_profile_and_search.sh \
  --model qwen3-8b \
  --branch codex/agentpipe-vp-search \
  --mode profile

bash experiments/bbt_16gpu/run_profile_and_search.sh \
  --model qwen3-8b \
  --branch codex/agentpipe-vp-search \
  --mode search \
  --trace-dir /workspace/runs/seampipe/qwen3-8b/profile-001/traces
```
