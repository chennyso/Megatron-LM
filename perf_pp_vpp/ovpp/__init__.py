"""OVPP: Overlap-guided Boundary Search for VPP Pipeline Parallelism.

Minimizes overlap deficit (exposed communication time) rather than just
balancing stage compute time.  Key insight: VPP's real value is creating
independent compute windows to hide recv wait, not balancing stages.
"""

from .profiler.layer_profiler import LayerProfiler, LayerProfile, SyntheticProfileGenerator
from .simulator.vpp_timeline_dag import VPPTimelineDAG, OverlapResult
from .search.overlap_guided_search import OverlapGuidedSearch, SearchConfig, SearchResult
from .visualization.ovpp_plots import OVPPVisualizer
