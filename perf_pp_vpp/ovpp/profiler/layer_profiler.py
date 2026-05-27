"""Per-layer profiling for OVPP: captures compute time, activation memory, and communication cost."""

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict


@dataclass
class LayerProfile:
    """Profile for a single transformer layer."""
    layer_id: int
    fwd_compute_ms: float   # forward compute time in ms
    bwd_compute_ms: float   # backward compute time in ms (typically 2x forward)
    activation_mb: float    # activation memory in MB
    recv_bytes: int = 0     # bytes received from previous stage (for comm modeling)
    send_bytes: int = 0     # bytes sent to next stage

    @property
    def total_compute_ms(self) -> float:
        return self.fwd_compute_ms + self.bwd_compute_ms


class LayerProfiler:
    """Profiles per-layer compute/memory/comm costs from nsys traces or logs."""

    def __init__(self, profile_path: Optional[str] = None):
        self.profiles: List[LayerProfile] = []
        if profile_path:
            self.load(profile_path)

    def load(self, path: str):
        """Load profiles from JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.profiles = [LayerProfile(**d) for d in data]

    def save(self, path: str):
        """Save profiles to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump([asdict(p) for p in self.profiles], f, indent=2)

    def add(self, profile: LayerProfile):
        self.profiles.append(profile)

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        return self.profiles[idx]

    def get_stage_time(self, layer_ids: List[int]) -> float:
        """Total fwd+bwd compute time for a set of layers."""
        return sum(self.profiles[i].total_compute_ms for i in layer_ids)

    def get_stage_fwd_time(self, layer_ids: List[int]) -> float:
        """Total forward compute time for a set of layers."""
        return sum(self.profiles[i].fwd_compute_ms for i in layer_ids)

    def get_stage_activation_mb(self, layer_ids: List[int]) -> float:
        """Total activation memory for a set of layers."""
        return sum(self.profiles[i].activation_mb for i in layer_ids)


class SyntheticProfileGenerator:
    """Generate synthetic layer profiles from model architecture config.

    Uses a roofline-inspired model:
      compute_time = FLOPs / peak_flops
      activation_memory = hidden * seq_len * bytes_per_element * factor
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        ffn_hidden_size: int = 25600,
        num_attention_heads: int = 64,
        num_kv_heads: int = 8,
        seq_length: int = 2048,
        dtype_bytes: int = 2,  # bf16 = 2 bytes
        peak_tflops: float = 312.0,  # A100 bf16 peak
        mem_bandwidth_gbs: float = 2.0,  # GB/s per GPU
    ):
        self.hidden = hidden_size
        self.ffn_hidden = ffn_hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.seq_len = seq_length
        self.dtype_bytes = dtype_bytes
        self.peak_tflops = peak_tflops
        self.mem_bw = mem_bandwidth_gbs

    def _attention_flops(self) -> int:
        """FLOPs for one attention layer (forward): 4 * seq^2 * hidden + 4 * seq * hidden^2."""
        h = self.hidden
        s = self.seq_len
        qkv_proj = 2 * s * h * (h + 2 * h * self.num_kv_heads // self.num_heads)
        attn_score = 2 * s * s * h
        out_proj = 2 * s * h * h
        return qkv_proj + attn_score + out_proj

    def _ffn_flops(self) -> int:
        """FLOPs for one FFN layer (forward): 2 * seq * (3 * hidden * ffn_hidden)."""
        return 2 * self.seq_len * 3 * self.hidden * self.ffn_hidden

    def _layer_flops(self, is_moe: bool = False, num_experts: int = 1, top_k: int = 1) -> int:
        attn = self._attention_flops()
        ffn = self._ffn_flops()
        if is_moe:
            # MoE: router + top_k experts, each expert is ffn/num_experts compute
            return attn + ffn * top_k
        return attn + ffn

    def _activation_memory_mb(self) -> float:
        """Activation memory per layer in MB (simplified)."""
        h = self.hidden
        s = self.seq_len
        # Attention: Q, K, V, O, softmax, dropout
        attn_act = 5 * s * h * self.dtype_bytes
        # FFN: two linear outputs + activation
        ffn_act = 3 * s * self.ffn_hidden * self.dtype_bytes
        # LayerNorm, residual
        misc = 4 * s * h * self.dtype_bytes
        return (attn_act + ffn_act + misc) / (1024 * 1024)

    def generate(
        self,
        num_layers: int = 64,
        is_moe_layers: Optional[List[int]] = None,
        num_experts: int = 1,
        top_k: int = 1,
        fwd_bwd_ratio: float = 2.5,
    ) -> LayerProfiler:
        """Generate synthetic profiles for all layers.

        Args:
            num_layers: Total transformer layers.
            is_moe_layers: List of layer indices that use MoE (None = dense only).
            num_experts: Number of experts for MoE layers.
            top_k: Top-k routing for MoE.
            fwd_bwd_ratio: backward_time / forward_time ratio.
        """
        if is_moe_layers is None:
            is_moe_layers = []

        profiler = LayerProfiler()
        for i in range(num_layers):
            is_moe = i in is_moe_layers
            flops = self._layer_flops(is_moe, num_experts, top_k)
            # Forward time: FLOPs / peak_flops (in ms)
            fwd_ms = flops / (self.peak_tflops * 1e12) * 1000
            # Memory-bound check: if FLOPs/byte < bandwidth, use memory-bound time
            bytes_moved = self._activation_memory_mb() * 1024 * 1024
            arithmetic_intensity = flops / bytes_moved if bytes_moved > 0 else 1e6
            roofline_intensity = self.peak_tflops * 1e12 / (self.mem_bw * 1e9)
            if arithmetic_intensity < roofline_intensity:
                fwd_ms = bytes_moved / (self.mem_bw * 1e9) * 1000

            bwd_ms = fwd_ms * fwd_bwd_ratio
            act_mb = self._activation_memory_mb()

            # Communication bytes: hidden * seq_len * dtype (activation send/recv)
            comm_bytes = self.hidden * self.seq_len * self.dtype_bytes

            profiler.add(LayerProfile(
                layer_id=i,
                fwd_compute_ms=round(fwd_ms, 4),
                bwd_compute_ms=round(bwd_ms, 4),
                activation_mb=round(act_mb, 2),
                recv_bytes=comm_bytes,
                send_bytes=comm_bytes,
            ))
        return profiler

    def generate_heterogeneous(
        self,
        num_layers: int = 64,
        num_moe_layers: int = 32,
        num_experts: int = 8,
        top_k: int = 2,
        fwd_bwd_ratio: float = 2.5,
        variation_pct: float = 0.15,
    ) -> LayerProfiler:
        """Generate heterogeneous profiles with MoE/dense variation and per-layer noise.

        This creates realistic non-uniform profiles where:
        - MoE layers are more expensive (top_k x dense FFN)
        - First/last layers have different costs (embedding/loss)
        - Random per-layer variation simulates real hardware behavior
        """
        import random
        random.seed(42)

        profiler = LayerProfiler()

        moe_stride = max(1, num_layers // num_moe_layers)
        moe_set = set(range(0, num_layers, moe_stride))

        for i in range(num_layers):
            is_moe = i in moe_set
            flops = self._layer_flops(is_moe, num_experts if is_moe else 1, top_k if is_moe else 1)

            fwd_ms = flops / (self.peak_tflops * 1e12) * 1000
            bytes_moved = self._activation_memory_mb() * 1024 * 1024
            arithmetic_intensity = flops / bytes_moved if bytes_moved > 0 else 1e6
            roofline_intensity = self.peak_tflops * 1e12 / (self.mem_bw * 1e9)
            if arithmetic_intensity < roofline_intensity:
                fwd_ms = bytes_moved / (self.mem_bw * 1e9) * 1000

            variation = 1.0 + random.uniform(-variation_pct, variation_pct)
            fwd_ms *= variation

            if i == 0:
                fwd_ms *= 1.1
            elif i == num_layers - 1:
                fwd_ms *= 1.05

            bwd_ms = fwd_ms * fwd_bwd_ratio
            act_mb = self._activation_memory_mb()

            comm_bytes = self.hidden * self.seq_len * self.dtype_bytes
            if is_moe:
                comm_bytes = int(comm_bytes * 1.2)

            profiler.add(LayerProfile(
                layer_id=i,
                fwd_compute_ms=round(fwd_ms, 4),
                bwd_compute_ms=round(bwd_ms, 4),
                activation_mb=round(act_mb, 2),
                recv_bytes=comm_bytes,
                send_bytes=comm_bytes,
            ))
        return profiler

    def generate_for_32b(self) -> LayerProfiler:
        """Convenience: generate profiles for a 32B model (64 layers, hidden=5120)."""
        return self.generate(
            num_layers=64,
            is_moe_layers=list(range(0, 64, 2)),  # every other layer is MoE
            num_experts=8,
            top_k=2,
        )
