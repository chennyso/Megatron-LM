"""OVPP Overlap-Guided Search: Simulated Annealing for optimal VPP chunk boundaries.

Three-phase search:
1. Balanced seed: start with equal-sized chunks
2. Boundary perturbation: randomly shift boundaries
3. SA local search: accept/reject based on Metropolis criterion

Optimization objective: min(overlap_deficit) — NOT min(bubble_time).
"""

import math
import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ovpp.simulator.vpp_timeline_dag import VPPTimelineDAG, OverlapResult


@dataclass
class SearchConfig:
    """Configuration for the SA search."""
    num_chunks: int = 4                    # number of VPP stages
    sa_initial_temp: float = 100.0         # SA initial temperature
    sa_cooling_rate: float = 0.95          # SA cooling schedule
    sa_min_temp: float = 0.01             # SA minimum temperature
    sa_max_iterations: int = 1000          # max iterations per temperature
    perturbation_range: int = 4            # max layers to shift a boundary
    min_chunk_size: int = 2                # minimum layers per chunk
    seed: int = 42                         # random seed for reproducibility
    time_limit_seconds: float = 300.0      # wall-clock time limit
    verbose: bool = False

    # Multi-objective weights
    deficit_weight: float = 1.0            # weight for overlap deficit
    bubble_weight: float = 0.1             # weight for bubble time (tiebreaker)


@dataclass
class SearchResult:
    """Result of the overlap-guided search."""
    best_boundaries: List[int]             # optimal chunk boundaries
    best_result: OverlapResult             # overlap analysis for best boundaries
    balanced_boundaries: List[int]         # initial balanced boundaries
    balanced_result: OverlapResult         # overlap analysis for balanced boundaries
    iterations: int                        # total SA iterations
    search_time_seconds: float             # wall-clock search time
    improvement_pct: float                 # deficit reduction vs balanced (%)
    trajectory: List[Tuple[List[int], float]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"OVPP Search Result:\n"
            f"  Balanced boundaries: {self.balanced_boundaries}\n"
            f"  Balanced deficit:    {self.balanced_result.total_overlap_deficit_ms:.2f} ms\n"
            f"  Optimal boundaries:  {self.best_boundaries}\n"
            f"  Optimal deficit:     {self.best_result.total_overlap_deficit_ms:.2f} ms\n"
            f"  Improvement:         {self.improvement_pct:.1f}%\n"
            f"  Iterations:          {self.iterations}\n"
            f"  Search time:         {self.search_time_seconds:.1f}s\n"
        )


class OverlapGuidedSearch:
    """Simulated Annealing search for optimal VPP chunk boundaries.

    The search optimizes min(overlap_deficit), meaning we find boundaries that
    maximize the overlap between communication and independent computation.

    This is fundamentally different from balanced partitioning (min stage time)
    or bubble-minimizing approaches (min idle time). The overlap deficit captures
    *exposed* communication — time when the network is busy but the GPU could
    be computing something else.
    """

    def __init__(self, dag_simulator: VPPTimelineDAG, config: Optional[SearchConfig] = None):
        self.dag = dag_simulator
        self.config = config or SearchConfig()
        random.seed(self.config.seed)

    def _objective(self, result: OverlapResult) -> float:
        """Weighted objective: deficit + bubble penalty."""
        return (
            self.config.deficit_weight * result.total_overlap_deficit_ms +
            self.config.bubble_weight * result.bubble_ms
        )

    def _generate_balanced_boundaries(self) -> List[int]:
        """Generate equal-sized chunk boundaries."""
        n = self.dag.num_layers
        k = self.config.num_chunks
        chunk_size = n // k
        boundaries = [i * chunk_size for i in range(k)]
        boundaries.append(n)
        return boundaries

    def _is_valid_boundaries(self, boundaries: List[int]) -> bool:
        """Check if boundaries satisfy constraints."""
        if boundaries[0] != 0 or boundaries[-1] != self.dag.num_layers:
            return False
        for i in range(1, len(boundaries)):
            if boundaries[i] - boundaries[i - 1] < self.config.min_chunk_size:
                return False
        # Must be sorted
        for i in range(len(boundaries) - 1):
            if boundaries[i] >= boundaries[i + 1]:
                return False
        return True

    def _perturb(self, boundaries: List[int]) -> List[int]:
        """Randomly perturb one internal boundary.

        We never move the first (0) or last (num_layers) boundary.
        """
        new_boundaries = list(boundaries)
        # Pick a random internal boundary (indices 1 to num_chunks-1)
        idx = random.randint(1, len(boundaries) - 2)
        max_shift = self.config.perturbation_range
        shift = random.randint(-max_shift, max_shift)

        new_boundaries[idx] = max(
            self.config.min_chunk_size,
            min(self.dag.num_layers - self.config.min_chunk_size * (len(boundaries) - idx - 1),
                new_boundaries[idx] + shift)
        )

        if self._is_valid_boundaries(new_boundaries):
            return new_boundaries
        return boundaries  # reject invalid

    def search(self) -> SearchResult:
        """Run the three-phase SA search.

        Phase 1: Evaluate balanced seed
        Phase 2: SA with boundary perturbation
        Phase 3: Return best found
        """
        start_time = time.time()
        cfg = self.config

        # Phase 1: Balanced seed
        balanced_bounds = self._generate_balanced_boundaries()
        balanced_result = self.dag.evaluate(balanced_bounds)
        balanced_obj = self._objective(balanced_result)

        # Initialize SA state
        current_bounds = list(balanced_bounds)
        current_result = balanced_result
        current_obj = balanced_obj

        best_bounds = list(balanced_bounds)
        best_result = balanced_result
        best_obj = balanced_obj

        trajectory = [(list(balanced_bounds), balanced_result.total_overlap_deficit_ms)]

        # Phase 2: SA search
        temp = cfg.sa_initial_temp
        iteration = 0

        while temp > cfg.sa_min_temp:
            if time.time() - start_time > cfg.time_limit_seconds:
                break

            for _ in range(cfg.sa_max_iterations):
                iteration += 1

                # Perturb
                new_bounds = self._perturb(current_bounds)
                if new_bounds == current_bounds:
                    continue

                # Evaluate
                new_result = self.dag.evaluate(new_bounds)
                new_obj = self._objective(new_result)

                # Metropolis acceptance
                delta = new_obj - current_obj
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_bounds = new_bounds
                    current_result = new_result
                    current_obj = new_obj

                    # Update best
                    if current_obj < best_obj:
                        best_bounds = list(current_bounds)
                        best_result = current_result
                        best_obj = current_obj
                        trajectory.append((list(best_bounds), best_result.total_overlap_deficit_ms))

                        if cfg.verbose:
                            print(f"  iter={iteration} temp={temp:.3f} "
                                  f"deficit={best_result.total_overlap_deficit_ms:.2f}ms "
                                  f"bounds={best_bounds}")

            temp *= cfg.sa_cooling_rate

        search_time = time.time() - start_time

        improvement = 0.0
        bal_deficit = balanced_result.total_overlap_deficit_ms
        best_deficit = best_result.total_overlap_deficit_ms
        if bal_deficit > 0:
            improvement = (bal_deficit - best_deficit) / bal_deficit * 100

        return SearchResult(
            best_boundaries=best_bounds,
            best_result=best_result,
            balanced_boundaries=balanced_bounds,
            balanced_result=balanced_result,
            iterations=iteration,
            search_time_seconds=search_time,
            improvement_pct=improvement,
            trajectory=trajectory,
        )

    def search_multi_seed(self, num_seeds: int = 5) -> SearchResult:
        """Run search with multiple random seeds and return best result."""
        best = None
        for seed in range(num_seeds):
            self.config.seed = seed
            random.seed(seed)
            result = self.search()
            if best is None or result.best_result.total_overlap_deficit_ms < best.best_result.total_overlap_deficit_ms:
                best = result
        return best

    def boundaries_to_layout_string(self, boundaries: List[int]) -> str:
        """Convert chunk boundaries to Megatron-LM layout string.

        Example: [0, 16, 32, 48, 64] with 64 layers →
            "E tttttttttttttttt|tttttttttttttttt|tttttttttttttttt|tttttttttttttttt L"

        Megatron format: E=embedding, t=transformer decoder, m=MTP, L=loss, |=stage separator
        """
        stages = []
        for i in range(len(boundaries) - 1):
            n_layers = boundaries[i + 1] - boundaries[i]
            stages.append('t' * n_layers)

        # Add embedding at start of first stage, loss at end of last stage
        layout = f"E {stages[0]}"
        for s in stages[1:]:
            layout += f"|{s}"
        layout += " L"
        return layout
