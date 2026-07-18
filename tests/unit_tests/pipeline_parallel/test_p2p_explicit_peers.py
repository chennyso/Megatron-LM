# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import call

import pytest
import torch

from megatron.core.pipeline_parallel.p2p_communication import (
    P2PCommunicator,
    PipelineP2PPeers,
    _batched_p2p_ops,
    _p2p_ops,
)


def test_batched_p2p_uses_operation_specific_peers(mocker):
    group = object()
    tensors = [object() for _ in range(4)]
    p2p_op = mocker.patch(
        "torch.distributed.P2POp",
        side_effect=lambda op, tensor, peer, process_group: (op, tensor, peer, process_group),
    )
    batch = mocker.patch("torch.distributed.batch_isend_irecv", return_value=[])
    peers = PipelineP2PPeers(
        send_forward=13,
        recv_forward=11,
        send_backward=7,
        recv_backward=17,
    )

    _batched_p2p_ops(
        tensor_send_prev=tensors[0],
        tensor_recv_prev=tensors[1],
        tensor_send_next=tensors[2],
        tensor_recv_next=tensors[3],
        group=group,
        prev_pipeline_rank=1,
        next_pipeline_rank=2,
        peers=peers,
    )

    assert [item.args[2] for item in p2p_op.call_args_list] == [7, 11, 13, 17]
    batch.assert_called_once()


@pytest.mark.parametrize("group_rank", [0, 1])
def test_unbatched_p2p_uses_operation_specific_peers(mocker, group_rank):
    group = SimpleNamespace(size=lambda: 4, rank=lambda: group_rank)
    tensors = [object() for _ in range(4)]
    mocker.patch("torch.distributed.get_backend", return_value="nccl")
    isend = mocker.patch("torch.distributed.isend", side_effect=lambda **kwargs: kwargs)
    irecv = mocker.patch("torch.distributed.irecv", side_effect=lambda **kwargs: kwargs)
    peers = PipelineP2PPeers(
        send_forward=13,
        recv_forward=11,
        send_backward=7,
        recv_backward=17,
    )

    _p2p_ops(
        tensor_send_prev=tensors[0],
        tensor_recv_prev=tensors[1],
        tensor_send_next=tensors[2],
        tensor_recv_next=tensors[3],
        group=group,
        prev_pipeline_rank=1,
        next_pipeline_rank=2,
        peers=peers,
    )

    assert {item.kwargs["dst"] for item in isend.call_args_list} == {7, 13}
    assert {item.kwargs["src"] for item in irecv.call_args_list} == {11, 17}


def test_explicit_peers_reject_ring_exchange_before_issuing_p2p():
    communicator = object.__new__(P2PCommunicator)
    communicator.config = SimpleNamespace(use_ring_exchange_p2p=True)
    communicator.next_rank = 2
    communicator.prev_rank = 0

    with pytest.raises(ValueError, match="incompatible with ring_exchange"):
        communicator._communicate(
            tensor_send_next=object(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            peers=PipelineP2PPeers(send_forward=3),
        )


def test_active_explicit_operation_requires_peer_rank():
    communicator = object.__new__(P2PCommunicator)
    communicator.config = SimpleNamespace(use_ring_exchange_p2p=False)
    communicator.next_rank = 2
    communicator.prev_rank = 0

    with pytest.raises(ValueError, match="send_forward"):
        communicator._communicate(
            tensor_send_next=object(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            peers=PipelineP2PPeers(),
        )


def test_send_forward_threads_explicit_peers_to_communicate(mocker):
    communicator = object.__new__(P2PCommunicator)
    communicator.config = SimpleNamespace(timers=None)
    communicate = mocker.patch.object(communicator, "_communicate")
    peers = PipelineP2PPeers(send_forward=13)
    tensor = torch.empty(1)

    communicator.send_forward(tensor, is_last_stage=False, peers=peers)

    communicate.assert_called_once_with(
        tensor_send_next=tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
        peers=peers,
    )


def test_map_pipeline_peers_converts_group_ranks_to_global_ranks(mocker):
    communicator = object.__new__(P2PCommunicator)
    communicator.pp_group = SimpleNamespace(size=lambda: 4)
    get_global_rank = mocker.patch(
        "torch.distributed.get_global_rank", side_effect=lambda group, rank: 10 + rank
    )

    peers = communicator.map_pipeline_peers(
        send_forward=3,
        recv_forward=1,
        send_backward=0,
        recv_backward=2,
    )

    assert peers == PipelineP2PPeers(
        send_forward=13,
        recv_forward=11,
        send_backward=10,
        recv_backward=12,
    )
    assert get_global_rank.call_args_list == [
        call(communicator.pp_group, rank) for rank in (3, 1, 0, 2)
    ]
