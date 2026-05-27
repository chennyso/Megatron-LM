"""OVPP Visualization: timeline, deficit breakdown, and boundary movement plots."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ovpp.simulator.vpp_timeline_dag import OverlapResult, TimelineEvent, OpType
from ovpp.search.overlap_guided_search import SearchResult


class OVPPVisualizer:
    """Generates ASCII and matplotlib visualizations for OVPP results."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_timeline_ascii(self, result: OverlapResult, num_devices: int, max_width: int = 80):
        """Print an ASCII timeline visualization."""
        if not result.timeline:
            print("No timeline events to display.")
            return

        # Find time range
        min_t = min(e.start_ms for e in result.timeline)
        max_t = max(e.end_ms for e in result.timeline)
        duration = max_t - min_t

        if duration == 0:
            print("Zero-duration timeline.")
            return

        # Group events by device
        device_events = {}
        for e in result.timeline:
            if e.device_id not in device_events:
                device_events[e.device_id] = []
            device_events[e.device_id].append(e)

        chars_per_ms = max_width / duration

        print(f"\n{'='*max_width}")
        print(f"VPP Timeline ({num_devices} devices, {duration:.1f}ms total)")
        print(f"{'='*max_width}")

        type_chars = {OpType.FWD: 'F', OpType.BWD: 'B', OpType.RECV: 'R', OpType.SEND: 'S', OpType.IDLE: '.'}

        for dev in sorted(device_events.keys()):
            line = ['.'] * max_width
            for e in device_events[dev]:
                start_col = int((e.start_ms - min_t) * chars_per_ms)
                end_col = int((e.end_ms - min_t) * chars_per_ms)
                char = type_chars.get(e.op_type, '?')
                for c in range(max(0, start_col), min(max_width, end_col)):
                    line[c] = char
            print(f"Dev {dev:2d} |{''.join(line)}|")

        print(f"{'='*max_width}")
        print(f"Legend: F=Forward, B=Backward, R=Recv, S=Send, .=Idle")
        print(f"Overlap deficit: {result.total_overlap_deficit_ms:.2f}ms")
        print(f"Bubble: {result.bubble_ms:.2f}ms")

    def print_deficit_breakdown(self, result: OverlapResult):
        """Print per-stage and per-comm deficit breakdown."""
        print(f"\n{'='*60}")
        print("Overlap Deficit Breakdown")
        print(f"{'='*60}")

        print(f"\nTotal overlap deficit: {result.total_overlap_deficit_ms:.2f} ms")
        print(f"Total compute:        {result.total_compute_ms:.2f} ms")
        print(f"Total communication:  {result.total_comm_ms:.2f} ms")
        print(f"Bubble:               {result.bubble_ms:.2f} ms")
        print(f"Makespan:             {result.makespan_ms:.2f} ms")
        print(f"Deficit ratio:        {result.deficit_ratio:.3f}")

        print(f"\nPer-stage deficit:")
        for i, d in enumerate(result.per_stage_deficit_ms):
            bar = '#' * int(d / max(result.per_stage_deficit_ms) * 30) if max(result.per_stage_deficit_ms) > 0 else ''
            print(f"  Stage {i}: {d:8.2f} ms  {bar}")

        print(f"\nPer-communication deficit:")
        for i, d in enumerate(result.per_comm_deficit_ms):
            bar = '#' * int(d / max(result.per_comm_deficit_ms) * 30) if max(result.per_comm_deficit_ms) > 0 else ''
            print(f"  Comm {i}->{i+1}: {d:8.2f} ms  {bar}")

    def print_search_summary(self, search_result: SearchResult):
        """Print search result summary."""
        print(search_result.summary())

        # Print boundary comparison
        print("Boundary comparison:")
        print(f"  Balanced: {search_result.balanced_boundaries}")
        print(f"  Optimal:  {search_result.best_boundaries}")

        # Print trajectory
        if search_result.trajectory:
            print(f"\nSearch trajectory ({len(search_result.trajectory)} improvements):")
            for i, (bounds, obj) in enumerate(search_result.trajectory):
                print(f"  Step {i}: bounds={bounds} deficit={obj:.2f}ms")

    def print_layout_string(self, boundaries: List[int], num_layers: int):
        """Print the Megatron-LM layout string for given boundaries."""
        stages = []
        for i in range(len(boundaries) - 1):
            n = boundaries[i + 1] - boundaries[i]
            stages.append('t' * n)

        layout = f"E {stages[0]}"
        for s in stages[1:]:
            layout += f"|{s}"
        layout += " L"

        print(f"\nMegatron-LM layout string:")
        print(f"  --pipeline-model-parallel-layout \"{layout}\"")
        print(f"  Stages: {len(stages)}, Layers per stage: {[len(s) for s in stages]}")

    def save_results_json(self, search_result: SearchResult, filename: str = "ovpp_results.json"):
        """Save results to JSON for later analysis."""
        data = {
            'balanced_boundaries': search_result.balanced_boundaries,
            'balanced_deficit_ms': search_result.balanced_result.total_overlap_deficit_ms,
            'optimal_boundaries': search_result.best_boundaries,
            'optimal_deficit_ms': search_result.best_result.total_overlap_deficit_ms,
            'improvement_pct': search_result.improvement_pct,
            'iterations': search_result.iterations,
            'search_time_s': search_result.search_time_seconds,
            'trajectory': search_result.trajectory,
        }
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")

    def plot_deficit_comparison(
        self,
        balanced: OverlapResult,
        optimal: OverlapResult,
        save_path: Optional[str] = None,
    ):
        """Plot deficit comparison bar chart (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("matplotlib not available, skipping plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Per-stage deficit
        stages = list(range(len(balanced.per_stage_deficit_ms)))
        x = range(len(stages))
        width = 0.35
        axes[0].bar([i - width/2 for i in x], balanced.per_stage_deficit_ms, width, label='Balanced')
        axes[0].bar([i + width/2 for i in x], optimal.per_stage_deficit_ms, width, label='OVPP')
        axes[0].set_xlabel('Stage')
        axes[0].set_ylabel('Deficit (ms)')
        axes[0].set_title('Per-Stage Overlap Deficit')
        axes[0].legend()

        # Overall comparison
        metrics = ['Total Deficit', 'Bubble', 'Compute']
        bal_vals = [balanced.total_overlap_deficit_ms, balanced.bubble_ms, balanced.total_compute_ms]
        opt_vals = [optimal.total_overlap_deficit_ms, optimal.bubble_ms, optimal.total_compute_ms]
        x2 = range(len(metrics))
        axes[1].bar([i - width/2 for i in x2], bal_vals, width, label='Balanced')
        axes[1].bar([i + width/2 for i in x2], opt_vals, width, label='OVPP')
        axes[1].set_xticks(list(x2))
        axes[1].set_xticklabels(metrics)
        axes[1].set_ylabel('Time (ms)')
        axes[1].set_title('Overall Comparison')
        axes[1].legend()

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "deficit_comparison.png"
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close()
