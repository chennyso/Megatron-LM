# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.pipeline_parallel.route_policy import (
    LogicalStage,
    P2PAction,
    PipelineRoute,
)


@pytest.mark.parametrize("virtual_chunks", [1, 2, 4, 8])
def test_standard_route_has_two_v_minus_one_crossings_on_two_nodes(virtual_chunks):
    route = PipelineRoute.standard(pipeline_size=8, virtual_chunks=virtual_chunks)
    node_by_rank = ("g5",) * 4 + ("g6",) * 4

    assert len(route.cross_node_edges(node_by_rank)) == 2 * virtual_chunks - 1


@pytest.mark.parametrize("virtual_chunks", [1, 2, 4, 8])
def test_folded_route_has_v_crossings_on_two_nodes(virtual_chunks):
    route = PipelineRoute.folded(pipeline_size=8, virtual_chunks=virtual_chunks)
    node_by_rank = ("g5",) * 4 + ("g6",) * 4

    assert len(route.cross_node_edges(node_by_rank)) == virtual_chunks


def test_folded_route_keeps_chunk_transitions_local_and_alternates_direction():
    route = PipelineRoute.folded(pipeline_size=4, virtual_chunks=3)

    assert route.stages == (
        LogicalStage(0, 0),
        LogicalStage(0, 1),
        LogicalStage(0, 2),
        LogicalStage(0, 3),
        LogicalStage(1, 3),
        LogicalStage(1, 2),
        LogicalStage(1, 1),
        LogicalStage(1, 0),
        LogicalStage(2, 0),
        LogicalStage(2, 1),
        LogicalStage(2, 2),
        LogicalStage(2, 3),
    )


@pytest.mark.parametrize("factory", [PipelineRoute.standard, PipelineRoute.folded])
def test_route_is_bijective_and_backward_is_exact_reverse(factory):
    route = factory(pipeline_size=4, virtual_chunks=3)

    route.verify()
    assert [(edge.source, edge.target) for edge in route.backward_edges] == [
        (edge.target, edge.source) for edge in reversed(route.forward_edges)
    ]


def test_remote_message_signatures_exclude_local_chunk_transition():
    route = PipelineRoute.folded(pipeline_size=2, virtual_chunks=2)

    forward = route.message_signatures("forward", "activation")
    backward = route.message_signatures("backward", "activation_grad")

    assert len(forward) == 2
    assert len(backward) == 2
    assert {(item.source_rank, item.target_rank) for item in forward} == {(0, 1), (1, 0)}
    assert {(item.source_rank, item.target_rank) for item in backward} == {(0, 1), (1, 0)}


def test_signature_verifier_rejects_missing_receive_and_wrong_peer():
    route = PipelineRoute.folded(pipeline_size=2, virtual_chunks=2)
    actions = route.communication_actions("forward", "activation")

    with pytest.raises(ValueError, match="do not match"):
        route.verify_send_recv_signatures("forward", "activation", actions[:-1])

    first = actions[0]
    wrong_peer = P2PAction(
        role=first.role,
        rank=first.rank,
        peer_rank=first.rank,
        signature=first.signature,
    )
    with pytest.raises(ValueError, match="send endpoints"):
        route.verify_send_recv_signatures(
            "forward", "activation", (wrong_peer,) + actions[1:]
        )


def test_route_rejects_duplicate_or_missing_stage():
    with pytest.raises(ValueError, match="bijection"):
        PipelineRoute(
            pipeline_size=2,
            virtual_chunks=2,
            stages=(
                LogicalStage(0, 0),
                LogicalStage(0, 1),
                LogicalStage(1, 0),
                LogicalStage(1, 0),
            ),
        )


def test_crossing_count_requires_complete_rank_topology():
    route = PipelineRoute.standard(pipeline_size=4, virtual_chunks=2)

    with pytest.raises(ValueError, match="one entry per physical pipeline rank"):
        route.cross_node_edges(("g5", "g6"))


@pytest.mark.parametrize("hierarchy_factor,expected_transitions", [(1, 15), (2, 7), (4, 3), (8, 1)])
def test_topology_segmented_route_realizes_intermediate_transition_budgets(
    hierarchy_factor, expected_transitions
):
    node_by_rank = ("g5",) * 4 + ("g6",) * 4
    route = PipelineRoute.topology_segmented(
        pipeline_size=8,
        virtual_chunks=8,
        node_by_rank=node_by_rank,
        hierarchy_factor=hierarchy_factor,
    )

    route.verify()
    assert len(route.cross_node_edges(node_by_rank)) == expected_transitions


def test_hierarchy_one_is_the_standard_route_on_contiguous_domains():
    node_by_rank = ("g5",) * 4 + ("g6",) * 4

    assert PipelineRoute.topology_segmented(
        pipeline_size=8,
        virtual_chunks=4,
        node_by_rank=node_by_rank,
        hierarchy_factor=1,
    ) == PipelineRoute.standard(pipeline_size=8, virtual_chunks=4)


def test_endpoint_rotation_spreads_transition_load_without_changing_budget():
    node_by_rank = ("g5",) * 4 + ("g6",) * 4
    fixed = PipelineRoute.topology_segmented(8, 8, node_by_rank, 2)
    rotated = PipelineRoute.topology_segmented(
        8, 8, node_by_rank, 2, rotate_endpoints=True
    )

    fixed_load = fixed.cross_node_endpoint_counts(node_by_rank)
    rotated_load = rotated.cross_node_endpoint_counts(node_by_rank)
    assert len(fixed.cross_node_edges(node_by_rank)) == len(
        rotated.cross_node_edges(node_by_rank)
    )
    assert max(rotated_load.values()) < max(fixed_load.values())


def test_transition_reduction_exposes_rank_reuse_tradeoff():
    node_by_rank = ("g5",) * 4 + ("g6",) * 4
    standard = PipelineRoute.topology_segmented(8, 8, node_by_rank, 1)
    one_transition = PipelineRoute.topology_segmented(8, 8, node_by_rank, 8)

    assert standard.rank_reuse_pressure() < one_transition.rank_reuse_pressure()
    assert min(gap for gaps in standard.rank_reuse_gaps().values() for gap in gaps) == 7
    assert min(gap for gaps in one_transition.rank_reuse_gaps().values() for gap in gaps) == 3


def test_topology_segmented_route_rejects_non_divisor_hierarchy():
    with pytest.raises(ValueError, match="positive divisor"):
        PipelineRoute.topology_segmented(
            pipeline_size=8,
            virtual_chunks=8,
            node_by_rank=("g5",) * 4 + ("g6",) * 4,
            hierarchy_factor=3,
        )
