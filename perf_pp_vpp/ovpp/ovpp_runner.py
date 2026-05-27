"""OVPP Runner: orchestrates profiling, simulation, search, and visualization.

Usage:
    python -m ovpp.ovpp_runner --num-layers 64 --num-chunks 4 --num-microbatches 8
    python -m ovpp.ovpp_runner --profile-path profiles.json --num-chunks 4
    python -m ovpp.ovpp_runner --synthetic --model-size 32b
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ovpp.profiler.layer_profiler import LayerProfiler, SyntheticProfileGenerator
from ovpp.simulator.vpp_timeline_dag import VPPTimelineDAG
from ovpp.search.overlap_guided_search import OverlapGuidedSearch, SearchConfig
from ovpp.visualization.ovpp_plots import OVPPVisualizer


def run_ovpp(
    profiler: LayerProfiler,
    num_chunks: int = 4,
    num_microbatches: int = 8,
    num_devices: int = 8,
    comm_bandwidth_gbps: float = 25.0,
    output_dir: str = "results",
    verbose: bool = False,
):
    """Run the full OVPP pipeline: simulate → search → visualize."""

    # Initialize DAG simulator
    dag = VPPTimelineDAG(
        profiler=profiler,
        num_devices=num_devices,
        num_microbatches=num_microbatches,
        comm_bandwidth_gbps=comm_bandwidth_gbps,
    )

    # Run search
    config = SearchConfig(
        num_chunks=num_chunks,
        verbose=verbose,
    )
    search = OverlapGuidedSearch(dag, config)
    result = search.search()

    # Visualize
    viz = OVPPVisualizer(output_dir)

    # Print results
    viz.print_search_summary(result)
    viz.print_deficit_breakdown(result.best_result)
    viz.print_timeline_ascii(result.best_result, num_devices)
    viz.print_layout_string(result.best_boundaries, len(profiler))

    # Save
    viz.save_results_json(result)
    viz.plot_deficit_comparison(result.balanced_result, result.best_result)

    return result


def main():
    parser = argparse.ArgumentParser(description="OVPP: Overlap-guided Boundary Search for VPP")
    parser.add_argument('--profile-path', type=str, default=None,
                        help='Path to layer profile JSON (from nsys)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic profiles')
    parser.add_argument('--model-size', type=str, default='32b',
                        choices=['7b', '14b', '32b', '70b'],
                        help='Model size for synthetic generation')
    parser.add_argument('--num-layers', type=int, default=64,
                        help='Number of transformer layers')
    parser.add_argument('--num-chunks', type=int, default=4,
                        help='Number of VPP chunks (virtual stages)')
    parser.add_argument('--num-microbatches', type=int, default=8,
                        help='Number of microbatches')
    parser.add_argument('--num-devices', type=int, default=8,
                        help='Number of GPU devices')
    parser.add_argument('--comm-bandwidth', type=float, default=25.0,
                        help='Inter-node communication bandwidth (GB/s)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Load or generate profiles
    if args.profile_path:
        print(f"Loading profiles from {args.profile_path}")
        profiler = LayerProfiler(args.profile_path)
    elif args.synthetic:
        print(f"Generating synthetic profiles for {args.model_size} model")
        model_configs = {
            '7b':  {'hidden': 4096,  'ffn': 11008, 'heads': 32, 'kv_heads': 32, 'layers': 32},
            '14b': {'hidden': 5120,  'ffn': 13824, 'heads': 40, 'kv_heads': 40, 'layers': 40},
            '32b': {'hidden': 5120,  'ffn': 25600, 'heads': 64, 'kv_heads': 8,  'layers': 64},
            '70b': {'hidden': 8192,  'ffn': 28672, 'heads': 64, 'kv_heads': 8,  'layers': 80},
        }
        cfg = model_configs[args.model_size]
        gen = SyntheticProfileGenerator(
            hidden_size=cfg['hidden'],
            ffn_hidden_size=cfg['ffn'],
            num_attention_heads=cfg['heads'],
            num_kv_heads=cfg['kv_heads'],
        )
        profiler = gen.generate_heterogeneous(
            num_layers=cfg['layers'],
            num_moe_layers=cfg['layers'] // 2,
            num_experts=8,
            top_k=2,
        )
    else:
        print("Error: provide --profile-path or --synthetic")
        parser.print_help()
        return

    print(f"Loaded {len(profiler)} layer profiles")

    # Run OVPP
    result = run_ovpp(
        profiler=profiler,
        num_chunks=args.num_chunks,
        num_microbatches=args.num_microbatches,
        num_devices=args.num_devices,
        comm_bandwidth_gbps=args.comm_bandwidth,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # Print Megatron command
    from ovpp.search.overlap_guided_search import OverlapGuidedSearch
    search = OverlapGuidedSearch(VPPTimelineDAG(profiler))
    layout = search.boundaries_to_layout_string(result.best_boundaries)
    print(f"\nTo use with Megatron-LM:")
    print(f"  --pipeline-model-parallel-layout \"{layout}\"")


if __name__ == '__main__':
    main()
