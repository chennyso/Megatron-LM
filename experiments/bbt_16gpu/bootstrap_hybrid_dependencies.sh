#!/usr/bin/env bash
set -euo pipefail

# Provision optional hybrid-model kernels on a BBT reservation Pod. The base
# NGC image has CUDA, PyTorch, Transformer Engine, and FlashAttention, but the
# native Mamba dependencies are intentionally not preinstalled. The public
# PyPI index can have an incomplete CA chain in these Pods, so package URLs are
# pinned directly and the local CUDA extensions are built from the checked-out
# sources rather than fetching an opaque prebuilt wheel.

ROOT="${1:-/workspace/runs/phaseweaver-20260818}"
DEPS="$ROOT/deps"
PYTHON_BIN="${PYTHON_BIN:-python}"

for dir in \
  "$DEPS/causal-conv1d-v1.5.3.post2-sm120" \
  "$DEPS/mamba-v2.2.6.post3-sm120"; do
  [[ -d "$dir" ]] || { echo "missing dependency source: $dir" >&2; exit 2; }
done

# Megatron's dataset helper is a local C++ extension.  Fresh checkouts do not
# contain the in-place build artifact, so build it before importing training
# modules; this is deterministic and avoids relying on a stale site package.
if [[ -f "$ROOT/code/megatron/core/datasets/helpers.cpp" ]]; then
  (cd "$ROOT/code" && "$PYTHON_BIN" setup.py build_ext --inplace)
elif [[ -f "$ROOT/megatron/core/datasets/helpers.cpp" ]]; then
  (cd "$ROOT" && "$PYTHON_BIN" setup.py build_ext --inplace)
fi

export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE

"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps \
  "$DEPS/causal-conv1d-v1.5.3.post2-sm120"
"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps \
  "$DEPS/mamba-v2.2.6.post3-sm120"

# These URLs and hashes are pinned in the repository's uv.lock.
env -u PIP_CONSTRAINT "$PYTHON_BIN" -m pip install --no-deps --trusted-host files.pythonhosted.org \
  https://files.pythonhosted.org/packages/ee/36/3c303f92bafea7c3f97d68bbb83d18cc42e30cd0bfb1b7cfe589360f11d6/fla_core-0.4.2-py3-none-any.whl \
  https://files.pythonhosted.org/packages/60/ee/a3cba17965482b35c4990af90bad108e82c32edcb59911c37f318b5f4198/flash_linear_attention-0.4.2-py3-none-any.whl
env -u PIP_CONSTRAINT "$PYTHON_BIN" -m pip install --no-cache-dir \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org transformers==5.8.0

"$PYTHON_BIN" - <<'PY'
from fla.layers.gated_deltanet import GatedDeltaNet
import causal_conv1d_cuda
import selective_scan_cuda

print("hybrid dependency check passed:", GatedDeltaNet.__name__)
PY
