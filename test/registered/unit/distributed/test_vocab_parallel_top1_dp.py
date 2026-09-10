"""_vocab_parallel_top1 must offset the winning local index by the
group-local rank (rank_in_group), not the global rank.

Under DP attention the logits are sharded over the attention-TP group; with
groups like [0, 1] / [2, 3], a member's global rank is not its vocab-shard
index. The bug is silent: a winner on group rank 0 of group [2, 3] (global
rank 2) lands at offset 2 * local_size instead of 0 — still inside the
tp_size * local_size bound, so it passes any range check (round-2 review N1).

CPU-only: the group is stubbed, and the peer rank's packed row is precomputed
with the correct rank_in_group convention so the test exercises only the rank
under test's offset math.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOCAL_SIZE = 4


class _StubGroup:
    def __init__(self, world_size, rank, rank_in_group, peer_row):
        self.world_size = world_size
        self.rank = rank
        self.rank_in_group = rank_in_group
        self._peer_row = peer_row

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        return torch.stack([self._peer_row, tensor[0]])


def _run_top1(monkeypatch, group, logits, dp_attention):
    module = pytest.importorskip(
        "sglang.srt.speculative.eagle_worker_v2"
    )
    import sglang.srt.distributed as dist_mod
    import sglang.srt.layers.dp_attention as dp_att_mod
    import sglang.srt.runtime_context as rc_mod

    monkeypatch.setattr(dist_mod, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        dp_att_mod, "is_dp_attention_enabled", lambda: dp_attention
    )
    parallel = type("P", (), {"attn_tp_group": group})()
    monkeypatch.setattr(rc_mod, "get_parallel", lambda: parallel)
    return module._vocab_parallel_top1(logits)


def test_winner_on_nonzero_group_rank_offsets_by_rank_in_group(monkeypatch):
    # Group [2, 3]: this rank is global 3 / group 1. It wins with local idx 1;
    # the peer (global 2 / group 0) reports val 0.5 at local idx 3.
    group = _StubGroup(2, rank=3, rank_in_group=1,
                       peer_row=torch.tensor([0.5, 3.0]))
    logits = torch.tensor([[0.1, 0.9, 0.2, 0.5]])
    _, idx = _run_top1(monkeypatch, group, logits, dp_attention=True)
    # Correct: 1 + 1 * 4. The global-rank bug would report 1 + 3 * 4 = 13.
    assert idx.item() == 5


def test_winner_on_group_rank_0_of_nonzero_global_rank(monkeypatch):
    # The silent-corruption case: this rank is global 2 / group 0 and wins
    # with local idx 0, so the correct global index is 0. The global-rank bug
    # reports 0 + 2 * 4 = 8 — wrong but in range for a 2 * 4 shard space.
    group = _StubGroup(2, rank=2, rank_in_group=0,
                       peer_row=torch.tensor([0.1, 6.0]))
    logits = torch.tensor([[0.9, 0.3, 0.1, 0.2]])
    _, idx = _run_top1(monkeypatch, group, logits, dp_attention=True)
    assert idx.item() == 0


def test_plain_tp_path_uses_group_rank_too(monkeypatch):
    # Without DP attention the full TP group is used; for a plain TP group
    # rank_in_group == global rank, so this pins the non-DP branch against
    # regression while keeping the same offset convention.
    group = _StubGroup(2, rank=1, rank_in_group=1,
                       peer_row=torch.tensor([0.5, 3.0]))
    logits = torch.tensor([[0.1, 0.9, 0.2, 0.5]])
    _, idx = _run_top1(monkeypatch, group, logits, dp_attention=False)
    assert idx.item() == 5


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v"]))
