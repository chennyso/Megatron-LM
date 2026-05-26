from __future__ import annotations

import functools
import os
from contextlib import contextmanager

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _nvtx_available() -> bool:
    return torch is not None and hasattr(torch.cuda, "nvtx")


@contextmanager
def nvtx_range(name: str):
    if _nvtx_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if _nvtx_available():
            torch.cuda.nvtx.range_pop()


def wrap_function(obj, attr: str, task_type: str):
    if not hasattr(obj, attr):
        return False
    fn = getattr(obj, attr)
    if getattr(fn, "_perf_pp_vpp_wrapped", False):
        return True

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rank = os.environ.get("RANK", "?")
        step = os.environ.get("PERF_STEP", "?")
        micro = os.environ.get("PERF_MICROBATCH", "?")
        vp = os.environ.get("VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK", os.environ.get("VIRTUAL_PIPELINE_RANK", "?"))
        name = f"task={task_type}|step={step}|microbatch={micro}|rank={rank}|vp={vp}"
        with nvtx_range(name):
            return fn(*args, **kwargs)

    wrapper._perf_pp_vpp_wrapped = True
    setattr(obj, attr, wrapper)
    return True


def install():
    try:
        from megatron.core.pipeline_parallel import schedules
        from megatron.core.pipeline_parallel import p2p_communication
    except Exception:
        return
    wrap_function(schedules, "forward_step", "F")
    wrap_function(schedules, "backward_step", "B")
    wrap_function(p2p_communication, "recv_forward", "RECV_F")
    wrap_function(p2p_communication, "send_forward", "SEND_F")
    wrap_function(p2p_communication, "recv_backward", "RECV_B")
    wrap_function(p2p_communication, "send_backward", "SEND_B")


install()
