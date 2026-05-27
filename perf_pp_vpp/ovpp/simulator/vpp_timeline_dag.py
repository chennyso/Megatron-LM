"""VPP Timeline DAG Simulator for OVPP.

Models the 1F1B schedule under Virtual Pipeline Parallelism and calculates
overlap deficit (exposed communication time) for each chunk boundary configuration.

Key insight: recv_time = waiting_for_upstream_compute (70-80%) + NCCL_stream +
                        TP_collective + actual_network_transfer
The VPP schedule creates independent compute windows that can hide recv wait.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ovpp.profiler.layer_profiler import LayerProfiler, LayerProfile


class OpType(Enum):
    FWD = "forward"
    BWD = "backward"
    RECV = "recv"
    SEND = "send"
    IDLE = "idle"


@dataclass
class TimelineEvent:
    """A single event on a device timeline."""
    device_id: int
    op_type: OpType
    start_ms: float
    duration_ms: float
    chunk_id: int = -1
    microbatch_id: int = -1
    layer_ids: List[int] = field(default_factory=list)

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


@dataclass
class OverlapResult:
    """Result of overlap analysis for one chunk boundary configuration."""
    total_overlap_deficit_ms: float   # sum of all exposed comm time
    per_stage_deficit_ms: List[float] # deficit per virtual stage
    per_comm_deficit_ms: List[float]  # deficit per communication edge
    total_compute_ms: float           # total compute time
    total_comm_ms: float              # total communication time
    bubble_ms: float                  # pipeline bubble time (idle)
    makespan_ms: float                # total execution time
    timeline: List[TimelineEvent] = field(default_factory=list)

    @property
    def deficit_ratio(self) -> float:
        """Deficit as fraction of total comm time."""
        return self.total_overlap_deficit_ms / self.total_comm_ms if self.total_comm_ms > 0 else 0


class VPPTimelineDAG:
    """Simulates VPP 1F1B schedule and computes overlap deficit.

    The 1F1B schedule for P stages and M microbatches:
    - Warmup: stage s processes fwd(mb=0..s) for s=0..P-1
    - Steady-state: for mb=P..M-1, each stage does fwd(mb) then bwd(mb-P)
    - Cooldown: stage s processes bwd(mb=M-P+s..M-1) for s=0..P-1

    Dependencies:
    - fwd(s, mb) requires fwd(s-1, mb) + comm (if s > 0)
    - bwd(s, mb) requires bwd(s+1, mb) + comm (if s < P-1)
    - Also: stage must complete prior ops in the schedule sequence
    """

    def __init__(
        self,
        profiler: LayerProfiler,
        num_devices: int = 8,
        num_microbatches: int = 8,
        comm_bandwidth_gbps: float = 25.0,
        intra_node_bandwidth_gbps: float = 50.0,
        tp_collective_ms: float = 0.5,
    ):
        self.profiler = profiler
        self.num_devices = num_devices
        self.num_microbatches = num_microbatches
        self.comm_bw = comm_bandwidth_gbps
        self.intra_bw = intra_node_bandwidth_gbps
        self.tp_collective_ms = tp_collective_ms
        self.num_layers = len(profiler)

    def _compute_stage_layers(self, chunk_boundaries: List[int]) -> List[List[int]]:
        """Convert chunk boundaries to layer assignments per stage."""
        stages = []
        for i in range(len(chunk_boundaries) - 1):
            start, end = chunk_boundaries[i], chunk_boundaries[i + 1]
            stages.append(list(range(start, end)))
        return stages

    def _comm_duration_ms(self, src_stage: int, dst_stage: int, data_bytes: int) -> float:
        """Communication duration between two stages."""
        if abs(src_stage - dst_stage) <= 1:
            bw = self.intra_bw
        else:
            bw = self.comm_bw
        return data_bytes / (bw * 1e9) * 1000

    def _generate_1f1b_schedule(self, num_stages: int, n_mb: int) -> List[Tuple[int, int, bool]]:
        """Generate the 1F1B schedule as a list of (stage, microbatch, is_forward) ops.

        The schedule follows Megatron-LM's 1F1B pattern:
        1. Warmup: increasing fwd passes per stage
        2. Steady: alternate fwd/bwd
        3. Cooldown: remaining bwd passes
        """
        schedule = []

        # Warmup: stage s does fwd for mb=0..s
        for s in range(num_stages):
            for mb in range(s + 1):
                schedule.append((s, mb, True))

        # Steady state: for mb=num_stages..n_mb-1, each stage does fwd(mb) then bwd(mb-num_stages)
        for mb in range(num_stages, n_mb):
            for s in range(num_stages):
                schedule.append((s, mb, True))
                schedule.append((s, mb - num_stages, False))

        # Cooldown: remaining bwd passes (reverse stage order)
        for s in range(num_stages - 1, -1, -1):
            for mb in range(n_mb - num_stages + s + 1, n_mb):
                schedule.append((s, mb, False))

        return schedule

    def _simulate_1f1b_schedule(
        self,
        stage_layers: List[List[int]],
    ) -> Tuple[List[TimelineEvent], Dict]:
        """Simulate 1F1B schedule for VPP.

        Returns timeline events and statistics.
        """
        num_stages = len(stage_layers)
        n_mb = self.num_microbatches
        events = []

        # Precompute fwd/bwd time per stage
        fwd_time = [self.profiler.get_stage_fwd_time(layers) for layers in stage_layers]
        bwd_time = [
            sum(self.profiler[i].bwd_compute_ms for i in layers)
            for layers in stage_layers
        ]

        # Comm time between adjacent stages
        comm_time = []
        for s in range(num_stages - 1):
            data_bytes = sum(self.profiler[i].send_bytes for i in stage_layers[s])
            comm_time.append(self._comm_duration_ms(s, s + 1, data_bytes))

        # Timing arrays: [stage][microbatch]
        fwd_start = [[float('inf')] * n_mb for _ in range(num_stages)]
        fwd_end = [[float('inf')] * n_mb for _ in range(num_stages)]
        bwd_start = [[float('inf')] * n_mb for _ in range(num_stages)]
        bwd_end = [[float('inf')] * n_mb for _ in range(num_stages)]

        # Per-stage busy-until tracker
        stage_free_at = [0.0] * num_stages

        # Generate and execute 1F1B schedule
        schedule = self._generate_1f1b_schedule(num_stages, n_mb)

        for s, mb, is_fwd in schedule:
            if is_fwd:
                # Forward: depends on stage_free_at[s] and (if s>0) fwd_end[s-1][mb] + comm
                earliest = stage_free_at[s]
                if s > 0 and fwd_end[s - 1][mb] < float('inf'):
                    earliest = max(earliest, fwd_end[s - 1][mb] + comm_time[s - 1])

                fwd_start[s][mb] = earliest
                fwd_end[s][mb] = earliest + fwd_time[s]
                stage_free_at[s] = fwd_end[s][mb]

                events.append(TimelineEvent(
                    device_id=s, op_type=OpType.FWD,
                    start_ms=earliest, duration_ms=fwd_time[s],
                    chunk_id=s, microbatch_id=mb, layer_ids=stage_layers[s],
                ))
            else:
                # Backward: depends on stage_free_at[s] and (if s<num_stages-1) bwd_end[s+1][mb] + comm
                earliest = stage_free_at[s]
                if s < num_stages - 1 and bwd_end[s + 1][mb] < float('inf'):
                    earliest = max(earliest, bwd_end[s + 1][mb] + comm_time[s])

                bwd_start[s][mb] = earliest
                bwd_end[s][mb] = earliest + bwd_time[s]
                stage_free_at[s] = bwd_end[s][mb]

                events.append(TimelineEvent(
                    device_id=s, op_type=OpType.BWD,
                    start_ms=earliest, duration_ms=bwd_time[s],
                    chunk_id=s, microbatch_id=mb, layer_ids=stage_layers[s],
                ))

        stats = {
            'fwd_start': fwd_start, 'fwd_end': fwd_end,
            'bwd_start': bwd_start, 'bwd_end': bwd_end,
            'comm_time': comm_time,
            'stage_layers': stage_layers,
            'stage_free_at': stage_free_at,
        }
        return events, stats

    def _compute_overlap_deficit(
        self,
        events: List[TimelineEvent],
        stats: Dict,
    ) -> Tuple[float, List[float], List[float], float, float, float, float]:
        """Compute overlap deficit for each communication edge.

        For each communication between stage s and s+1:
        - Forward: stage s sends activation after fwd[s][mb], stage s+1 needs it before fwd[s+1][mb]
        - Overlap window = time s+1 is doing independent compute while comm is in flight
        - Deficit = max(0, comm_duration - overlap_window)

        The overlap window for forward comm:
          = time between send_start (fwd_end[s][mb]) and when s+1 needs the data (fwd_start[s+1][mb])
          if s+1 is doing other work during this window, comm is hidden
          deficit = max(0, comm - (fwd_start[s+1][mb] - fwd_end[s][mb]))
          This is non-zero when s+1 needs the data BEFORE comm finishes.

        Similarly for backward comm.
        """
        stage_layers = stats['stage_layers']
        num_stages = len(stage_layers)
        comm_time = stats['comm_time']
        fwd_start = stats['fwd_start']
        fwd_end = stats['fwd_end']
        bwd_start = stats['bwd_start']
        bwd_end = stats['bwd_end']
        n_mb = self.num_microbatches

        per_comm_deficit = []
        per_stage_deficit = [0.0] * num_stages

        for s in range(num_stages - 1):
            ct = comm_time[s]
            total_deficit = 0.0

            for mb in range(n_mb):
                # Forward comm: stage s -> stage s+1
                if fwd_end[s][mb] < float('inf') and fwd_start[s + 1][mb] < float('inf'):
                    # Time gap between send completion and when recv is needed
                    gap = fwd_start[s + 1][mb] - fwd_end[s][mb]
                    # Deficit: comm exceeds the gap (i.e., s+1 has to wait for data)
                    deficit_fwd = max(0, ct - max(0, gap))
                    total_deficit += deficit_fwd

                # Backward comm: stage s+1 -> stage s
                if bwd_end[s + 1][mb] < float('inf') and bwd_start[s][mb] < float('inf'):
                    gap_bwd = bwd_start[s][mb] - bwd_end[s + 1][mb]
                    deficit_bwd = max(0, ct - max(0, gap_bwd))
                    total_deficit += deficit_bwd

            per_comm_deficit.append(total_deficit)
            per_stage_deficit[s] += total_deficit / 2
            per_stage_deficit[s + 1] += total_deficit / 2

        total_deficit = sum(per_comm_deficit)

        # Compute total compute time
        total_compute = n_mb * sum(fwd_time + bwd_time for fwd_time, bwd_time in zip(
            [self.profiler.get_stage_fwd_time(layers) for layers in stage_layers],
            [sum(self.profiler[i].bwd_compute_ms for i in layers) for layers in stage_layers]
        ))

        # Makespan and bubble
        makespan = max(stats['stage_free_at'])
        bubble = max(0, makespan * num_stages - total_compute)

        # Total comm time
        total_comm = sum(ct * n_mb * 2 for ct in comm_time)

        return total_deficit, per_stage_deficit, per_comm_deficit, total_compute, total_comm, bubble, makespan

    def evaluate(self, chunk_boundaries: List[int]) -> OverlapResult:
        """Evaluate a chunk boundary configuration.

        Args:
            chunk_boundaries: sorted list of boundary indices, e.g., [0, 16, 32, 48, 64]
                for 4 stages with 16 layers each.

        Returns:
            OverlapResult with deficit analysis.
        """
        stage_layers = self._compute_stage_layers(chunk_boundaries)
        events, stats = self._simulate_1f1b_schedule(stage_layers)
        deficit, per_stage, per_comm, total_compute, total_comm, bubble, makespan = \
            self._compute_overlap_deficit(events, stats)

        return OverlapResult(
            total_overlap_deficit_ms=deficit,
            per_stage_deficit_ms=per_stage,
            per_comm_deficit_ms=per_comm,
            total_compute_ms=total_compute,
            total_comm_ms=total_comm,
            bubble_ms=bubble,
            makespan_ms=makespan,
            timeline=events,
        )

    def evaluate_balanced(self, num_chunks: int) -> OverlapResult:
        """Evaluate balanced (equal-sized) chunk boundaries."""
        layers_per_chunk = self.num_layers // num_chunks
        boundaries = [i * layers_per_chunk for i in range(num_chunks)] + [self.num_layers]
        return self.evaluate(boundaries)
