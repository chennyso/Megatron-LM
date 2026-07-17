# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logical pipeline routes and static communication verification."""

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence


@dataclass(frozen=True, order=True)
class LogicalStage:
    """One virtual model chunk resident on a physical pipeline rank."""

    virtual_chunk: int
    physical_rank: int


@dataclass(frozen=True)
class PipelineEdge:
    """A dependency between adjacent stages in logical model order."""

    edge_id: int
    source: LogicalStage
    target: LogicalStage


@dataclass(frozen=True)
class MessageSignature:
    """Static identity used to match a send with exactly one receive."""

    edge_id: int
    direction: str
    tensor_kind: str
    source_rank: int
    target_rank: int


@dataclass(frozen=True)
class P2PAction:
    """One endpoint action produced when a route edge is lowered to P2P."""

    role: str
    rank: int
    peer_rank: int
    signature: MessageSignature


@dataclass(frozen=True)
class PipelineRoute:
    """A total logical ordering of all virtual stages in a pipeline."""

    pipeline_size: int
    virtual_chunks: int
    stages: tuple[LogicalStage, ...]

    def __post_init__(self) -> None:
        if self.pipeline_size < 1:
            raise ValueError("pipeline_size must be positive")
        if self.virtual_chunks < 1:
            raise ValueError("virtual_chunks must be positive")
        self.verify_bijection()

    @classmethod
    def standard(cls, pipeline_size: int, virtual_chunks: int) -> "PipelineRoute":
        """Build Megatron's conventional rank-increasing route per chunk."""
        stages = tuple(
            LogicalStage(virtual_chunk, physical_rank)
            for virtual_chunk in range(virtual_chunks)
            for physical_rank in range(pipeline_size)
        )
        return cls(pipeline_size, virtual_chunks, stages)

    @classmethod
    def folded(cls, pipeline_size: int, virtual_chunks: int) -> "PipelineRoute":
        """Build a serpentine route that alternates rank direction by chunk."""
        stages = []
        for virtual_chunk in range(virtual_chunks):
            physical_ranks: Iterable[int]
            if virtual_chunk % 2 == 0:
                physical_ranks = range(pipeline_size)
            else:
                physical_ranks = range(pipeline_size - 1, -1, -1)
            stages.extend(LogicalStage(virtual_chunk, rank) for rank in physical_ranks)
        return cls(pipeline_size, virtual_chunks, tuple(stages))

    def verify_bijection(self) -> None:
        """Require every (virtual chunk, physical rank) pair exactly once."""
        expected = {
            LogicalStage(virtual_chunk, physical_rank)
            for virtual_chunk in range(self.virtual_chunks)
            for physical_rank in range(self.pipeline_size)
        }
        actual = Counter(self.stages)
        duplicates = sorted(stage for stage, count in actual.items() if count != 1)
        missing = sorted(expected.difference(actual))
        unexpected = sorted(set(actual).difference(expected))
        if len(self.stages) != len(expected) or duplicates or missing or unexpected:
            raise ValueError(
                "route must be a bijection over virtual chunks and physical ranks; "
                f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
            )

    @property
    def forward_edges(self) -> tuple[PipelineEdge, ...]:
        """Return adjacent dependencies in forward model order."""
        return tuple(
            PipelineEdge(edge_id, source, target)
            for edge_id, (source, target) in enumerate(zip(self.stages, self.stages[1:]))
        )

    @property
    def backward_edges(self) -> tuple[PipelineEdge, ...]:
        """Return the exact reverse dependencies used by backpropagation."""
        return tuple(
            PipelineEdge(edge.edge_id, edge.target, edge.source)
            for edge in reversed(self.forward_edges)
        )

    def predecessor(self, stage: LogicalStage) -> LogicalStage | None:
        """Return the previous logical stage, including local chunk transitions."""
        index = self.stages.index(stage)
        return self.stages[index - 1] if index > 0 else None

    def successor(self, stage: LogicalStage) -> LogicalStage | None:
        """Return the next logical stage, including local chunk transitions."""
        index = self.stages.index(stage)
        return self.stages[index + 1] if index + 1 < len(self.stages) else None

    def cross_node_edges(self, node_by_rank: Sequence[Hashable]) -> tuple[PipelineEdge, ...]:
        """Return forward edges whose physical endpoints are on different nodes."""
        if len(node_by_rank) != self.pipeline_size:
            raise ValueError("node_by_rank must contain one entry per physical pipeline rank")
        return tuple(
            edge
            for edge in self.forward_edges
            if node_by_rank[edge.source.physical_rank]
            != node_by_rank[edge.target.physical_rank]
        )

    def message_signatures(
        self, direction: str, tensor_kind: str
    ) -> tuple[MessageSignature, ...]:
        """Lower route edges to direction-aware point-to-point message identities."""
        if direction == "forward":
            edges = self.forward_edges
        elif direction == "backward":
            edges = self.backward_edges
        else:
            raise ValueError("direction must be 'forward' or 'backward'")
        return tuple(
            MessageSignature(
                edge_id=edge.edge_id,
                direction=direction,
                tensor_kind=tensor_kind,
                source_rank=edge.source.physical_rank,
                target_rank=edge.target.physical_rank,
            )
            for edge in edges
            if edge.source.physical_rank != edge.target.physical_rank
        )

    def verify_backward_is_reverse(self) -> None:
        """Check that every backward edge reverses the corresponding forward edge."""
        forward_by_id = {edge.edge_id: edge for edge in self.forward_edges}
        for backward in self.backward_edges:
            forward = forward_by_id[backward.edge_id]
            if (backward.source, backward.target) != (forward.target, forward.source):
                raise ValueError(f"backward edge {backward.edge_id} does not reverse forward")

    def communication_actions(
        self, direction: str, tensor_kind: str
    ) -> tuple[P2PAction, ...]:
        """Lower every remote dependency to an explicit send and receive action."""
        actions = []
        for signature in self.message_signatures(direction, tensor_kind):
            actions.extend(
                (
                    P2PAction(
                        role="send",
                        rank=signature.source_rank,
                        peer_rank=signature.target_rank,
                        signature=signature,
                    ),
                    P2PAction(
                        role="recv",
                        rank=signature.target_rank,
                        peer_rank=signature.source_rank,
                        signature=signature,
                    ),
                )
            )
        return tuple(actions)

    def verify_send_recv_signatures(
        self,
        direction: str,
        tensor_kind: str,
        actions: Sequence[P2PAction] | None = None,
    ) -> None:
        """Require one endpoint-correct send and receive for every remote edge."""
        expected = set(self.message_signatures(direction, tensor_kind))
        actions = self.communication_actions(direction, tensor_kind) if actions is None else actions
        sends: Counter[MessageSignature] = Counter()
        receives: Counter[MessageSignature] = Counter()
        for action in actions:
            signature = action.signature
            if signature not in expected:
                raise ValueError(f"unexpected message signature: {signature}")
            if action.role == "send":
                if (action.rank, action.peer_rank) != (
                    signature.source_rank,
                    signature.target_rank,
                ):
                    raise ValueError(f"send endpoints do not match signature: {action}")
                sends[signature] += 1
            elif action.role == "recv":
                if (action.rank, action.peer_rank) != (
                    signature.target_rank,
                    signature.source_rank,
                ):
                    raise ValueError(f"receive endpoints do not match signature: {action}")
                receives[signature] += 1
            else:
                raise ValueError(f"unknown P2P action role: {action.role}")
        expected_counts = Counter({signature: 1 for signature in expected})
        if sends != expected_counts or receives != expected_counts:
            raise ValueError(
                "send and receive signatures do not match; "
                f"expected={expected_counts}, sends={sends}, receives={receives}"
            )

    def verify(self) -> None:
        """Run all topology-independent static route checks."""
        self.verify_bijection()
        self.verify_backward_is_reverse()
        self.verify_send_recv_signatures("forward", "activation")
        self.verify_send_recv_signatures("backward", "activation_grad")
