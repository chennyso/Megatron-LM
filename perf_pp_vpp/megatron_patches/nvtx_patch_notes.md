# NVTX patch notes

This patch path is intentionally reversible and low-intrusion.

- The runtime entrypoint imports `perf_pp_vpp.megatron_patches.nvtx_instrumentation`
- `nvtx_instrumentation.install()` monkey-patches selected Megatron pipeline functions at import time
- `patch_megatron_nvtx.py` can also inject or remove a guarded bootstrap import in `pretrain_gpt.py`

Current coverage:

- pipeline schedule forward/backward wrappers
- p2p send / recv wrappers where function names are discoverable
- generic rank / microbatch / step context tags

Current limitation:

- `B_IN` and `B_W` are not always cleanly separable without deeper autograd instrumentation
- task labels are best-effort and may need branch-specific adjustment
