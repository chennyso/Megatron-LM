# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Logical pipeline routes and static communication verification."""

from __future__ import annotations

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


@dataclass(frozen=True, order=True)
class RouteEndpointKey:
    """Identity of one stage-local communication operation."""

    stage: LogicalStage
    direction: str
    role: str


@dataclass(frozen=True)
class RoutePeer:
    """Resolved dependency for one route endpoint.

    ``kind`` distinguishes a model boundary, a same-rank dependency, and a
    dependency that must be lowered to distributed P2P. Only remote entries
    carry a message signature.
    """

    key: RouteEndpointKey
    kind: str
    edge_id: int | None
    peer_stage: LogicalStage | None
    peer_rank: int | None
    signature: MessageSignature | None


@dataclass(frozen=True)
class LayerAssignment:
    """Contiguous global layer interval assigned to one logical stage."""

    stage: LogicalStage
    start_layer: int
    end_layer: int

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer


class RoutePeerMap:
    """Deterministically lower logical route dependencies to stage-local peers."""

    _TENSOR_KIND = {
        "forward": "activation",
        "backward": "activation_grad",
    }

    def __init__(self, route: "PipelineRoute") -> None:
        self.route = route
        self._position_by_stage = {
            stage: position for position, stage in enumerate(route.stages)
        }
        self._edge_by_pair = {
            (edge.source, edge.target): edge for edge in route.forward_edges
        }
        self._peers = {
            RouteEndpointKey(stage, direction, role): self._resolve(
                stage, direction, role
            )
            for stage in route.stages
            for direction in ("forward", "backward")
            for role in ("send", "recv")
        }
        self.verify()

    def _neighbor(
        self, stage: LogicalStage, direction: str, role: str
    ) -> LogicalStage | None:
        if direction not in self._TENSOR_KIND:
            raise ValueError("direction must be 'forward' or 'backward'")
        if role not in ("send", "recv"):
            raise ValueError("role must be 'send' or 'recv'")
        follows_model_order = (direction, role) in (
            ("forward", "send"),
            ("backward", "recv"),
        )
        position = self._position_by_stage[stage]
        peer_position = position + 1 if follows_model_order else position - 1
        if peer_position < 0 or peer_position >= len(self.route.stages):
            return None
        return self.route.stages[peer_position]

    def _resolve(self, stage: LogicalStage, direction: str, role: str) -> RoutePeer:
        key = RouteEndpointKey(stage, direction, role)
        peer_stage = self._neighbor(stage, direction, role)
        if peer_stage is None:
            return RoutePeer(key, "terminal", None, None, None, None)

        if self._position_by_stage[stage] < self._position_by_stage[peer_stage]:
            forward_pair = (stage, peer_stage)
        else:
            forward_pair = (peer_stage, stage)
        edge = self._edge_by_pair[forward_pair]
        if peer_stage.physical_rank == stage.physical_rank:
            return RoutePeer(
                key,
                "local",
                edge.edge_id,
                peer_stage,
                peer_stage.physical_rank,
                None,
            )

        if role == "send":
            source_rank, target_rank = stage.physical_rank, peer_stage.physical_rank
        else:
            source_rank, target_rank = peer_stage.physical_rank, stage.physical_rank
        signature = MessageSignature(
            edge_id=edge.edge_id,
            direction=direction,
            tensor_kind=self._TENSOR_KIND[direction],
            source_rank=source_rank,
            target_rank=target_rank,
        )
        return RoutePeer(
            key,
            "remote",
            edge.edge_id,
            peer_stage,
            peer_stage.physical_rank,
            signature,
        )

    def get(
        self, stage: LogicalStage, direction: str, role: str
    ) -> RoutePeer:
        """Return the resolved endpoint dependency for one logical stage."""
        key = RouteEndpointKey(stage, direction, role)
        try:
            return self._peers[key]
        except KeyError as error:
            if stage not in self.route.stages:
                raise ValueError(f"stage is not present in route: {stage}") from error
            raise ValueError(f"invalid route endpoint key: {key}") from error

    def lower_group(
        self, keys: Sequence[RouteEndpointKey]
    ) -> tuple[RoutePeer, ...]:
        """Lower a combined communication call without assuming one shared peer."""
        if len(set(keys)) != len(keys):
            raise ValueError("combined communication group contains duplicate operations")
        return tuple(self.get(key.stage, key.direction, key.role) for key in keys)

    def verify(self) -> None:
        """Check boundary, locality, and exact forward/backward edge reversal."""
        allowed_kinds = {"terminal", "local", "remote"}
        for key, peer in self._peers.items():
            if peer.kind not in allowed_kinds:
                raise ValueError(f"unknown route peer kind: {peer.kind}")
            if key.stage.physical_rank != peer.key.stage.physical_rank:
                raise ValueError(f"route peer key changed physical rank: {peer}")
            if peer.kind == "terminal":
                if any(
                    item is not None
                    for item in (peer.edge_id, peer.peer_stage, peer.peer_rank, peer.signature)
                ):
                    raise ValueError(f"terminal endpoint carries a dependency: {peer}")
                continue
            if peer.peer_stage is None or peer.edge_id is None:
                raise ValueError(f"non-terminal endpoint lacks an edge: {peer}")
            if peer.peer_rank != peer.peer_stage.physical_rank:
                raise ValueError(f"peer rank does not match peer stage: {peer}")
            if peer.kind == "local" and peer.signature is not None:
                raise ValueError(f"local dependency must not emit P2P: {peer}")
            if peer.kind == "remote":
                if peer.signature is None:
                    raise ValueError(f"remote dependency lacks a signature: {peer}")
                if peer.peer_rank == key.stage.physical_rank:
                    raise ValueError(f"remote dependency resolves to the local rank: {peer}")

        for edge in self.route.forward_edges:
            forward_send = self.get(edge.source, "forward", "send")
            forward_recv = self.get(edge.target, "forward", "recv")
            backward_send = self.get(edge.target, "backward", "send")
            backward_recv = self.get(edge.source, "backward", "recv")
            kinds = {
                forward_send.kind,
                forward_recv.kind,
                backward_send.kind,
                backward_recv.kind,
            }
            expected_kind = (
                "local"
                if edge.source.physical_rank == edge.target.physical_rank
                else "remote"
            )
            if kinds != {expected_kind}:
                raise ValueError(f"edge locality is inconsistent for edge {edge.edge_id}")
            if expected_kind == "remote":
                if forward_send.signature != forward_recv.signature:
                    raise ValueError(f"forward signatures disagree for edge {edge.edge_id}")
                if backward_send.signature != backward_recv.signature:
                    raise ValueError(f"backward signatures disagree for edge {edge.edge_id}")
                forward_signature = forward_send.signature
                backward_signature = backward_send.signature
                if (
                    forward_signature.source_rank,
                    forward_signature.target_rank,
                ) != (
                    backward_signature.target_rank,
                    backward_signature.source_rank,
                ):
                    raise ValueError(f"backward P2P does not reverse edge {edge.edge_id}")


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

    @classmethod
    def topology_segmented(
        cls,
        pipeline_size: int,
        virtual_chunks: int,
        node_by_rank: Sequence[Hashable],
        hierarchy_factor: int,
        *,
        rotate_endpoints: bool = False,
    ) -> "PipelineRoute":
        """Group h virtual chunks inside each topology-domain run.

        For two domains and ``h | V``, the forward route has ``2V/h - 1``
        cross-domain transitions. Optional per-group rank rotation preserves
        that budget while spreading transition endpoints across local ranks.
        """
        if len(node_by_rank) != pipeline_size:
            raise ValueError("node_by_rank must contain one entry per physical pipeline rank")
        if hierarchy_factor < 1 or virtual_chunks % hierarchy_factor != 0:
            raise ValueError("hierarchy_factor must be a positive divisor of virtual_chunks")

        ranks_by_domain: dict[Hashable, list[int]] = {}
        for rank, domain in enumerate(node_by_rank):
            ranks_by_domain.setdefault(domain, []).append(rank)
        if not ranks_by_domain:
            raise ValueError("node_by_rank must define at least one topology domain")

        stages: list[LogicalStage] = []
        for group_index, first_chunk in enumerate(
            range(0, virtual_chunks, hierarchy_factor)
        ):
            chunks = range(first_chunk, first_chunk + hierarchy_factor)
            for domain_ranks in ranks_by_domain.values():
                rotation = group_index % len(domain_ranks) if rotate_endpoints else 0
                ordered_ranks = domain_ranks[rotation:] + domain_ranks[:rotation]
                for virtual_chunk in chunks:
                    stages.extend(
                        LogicalStage(virtual_chunk, physical_rank)
                        for physical_rank in ordered_ranks
                    )
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

    @property
    def first_stage(self) -> LogicalStage:
        """Return the route-defined model input stage."""
        return self.stages[0]

    @property
    def last_stage(self) -> LogicalStage:
        """Return the route-defined loss/output stage."""
        return self.stages[-1]

    def is_first_stage(self, stage: LogicalStage) -> bool:
        """Return whether a logical stage owns the model input boundary."""
        return stage == self.first_stage

    def is_last_stage(self, stage: LogicalStage) -> bool:
        """Return whether a logical stage owns the model output boundary."""
        return stage == self.last_stage

    def peer_map(self) -> RoutePeerMap:
        """Build the verified stage-local communication map for this route."""
        return RoutePeerMap(self)

    def assign_layers(self, layer_counts: Sequence[int]) -> tuple[LayerAssignment, ...]:
        """Assign contiguous model layers in route order using explicit cut sizes."""
        if len(layer_counts) != len(self.stages):
            raise ValueError("layer_counts must contain one entry per logical stage")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in layer_counts
        ):
            raise ValueError("each logical stage must receive a positive integer layer count")

        assignments = []
        start_layer = 0
        for stage, count in zip(self.stages, layer_counts):
            assignments.append(LayerAssignment(stage, start_layer, start_layer + count))
            start_layer += count
        return tuple(assignments)

    def assign_balanced_layers(self, num_layers: int) -> tuple[LayerAssignment, ...]:
        """Generate a deterministic near-uniform layout in logical route order."""
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < len(
            self.stages
        ):
            raise ValueError("num_layers must provide at least one layer per logical stage")
        quotient, remainder = divmod(num_layers, len(self.stages))
        counts = tuple(
            quotient + (position < remainder) for position in range(len(self.stages))
        )
        return self.assign_layers(counts)

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

    def cross_node_endpoint_counts(
        self, node_by_rank: Sequence[Hashable]
    ) -> Counter[int]:
        """Count cross-domain forward-edge incidences at each physical rank."""
        counts: Counter[int] = Counter()
        for edge in self.cross_node_edges(node_by_rank):
            counts[edge.source.physical_rank] += 1
            counts[edge.target.physical_rank] += 1
        return counts

    def rank_reuse_gaps(self) -> dict[int, tuple[int, ...]]:
        """Return intervening-stage counts between uses of each physical rank."""
        positions: dict[int, list[int]] = {
            rank: [] for rank in range(self.pipeline_size)
        }
        for position, stage in enumerate(self.stages):
            positions[stage.physical_rank].append(position)
        return {
            rank: tuple(right - left - 1 for left, right in zip(items, items[1:]))
            for rank, items in positions.items()
        }

    def rank_reuse_pressure(self) -> float:
        """Summarize short physical-rank reuse intervals; lower is better."""
        gaps = [gap for rank_gaps in self.rank_reuse_gaps().values() for gap in rank_gaps]
        if not gaps:
            return 0.0
        return sum(1.0 / (gap + 1.0) for gap in gaps) / len(gaps)

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
        self.peer_map().verify()
