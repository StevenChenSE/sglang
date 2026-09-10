"""Unit tests for the DFLASH M10 fix (round-3 review):

1. An extend batch with active multimodal inputs takes the SAME path as text:
   the target forward runs with CaptureHiddenMode.FULL and the chunk is
   materialized into the draft KV via _append_target_hidden_to_draft_kv_by_loc
   (previously the MM branch returned early with NULL capture, leaving a
   permanent draft-KV hole for image chunks).
2. SGLANG_DFLASH_MM_SKIP_DRAFT_KV=1 restores the old bypass (NULL capture, no
   materialization) as an escape hatch.
3. The decode-mode MM fallback scopes its DECODE-shaped view of the scheduler's
   batch to the target call: forward_mode / input_ids / out_cache_loc are
   restored after the call, and spec_info is never leaked.

Everything runs on CPU with stub worker attributes; no model is loaded.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative import dflash_worker_v2 as dwv
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class _SeqLens:
    """Tensor-ish stub: len(), +, record_stream (CPU-safe)."""

    def __init__(self, lens):
        self._lens = list(lens)

    def __len__(self):
        return len(self._lens)

    def numel(self):
        return len(self._lens)

    def to(self, dtype=None, device=None):
        return torch.tensor(self._lens, dtype=torch.int64)

    def __add__(self, other):
        return _SeqLens([x + other for x in self._lens])

    def record_stream(self, stream):
        pass


def _extend_batch(mm: bool, with_spec_info: bool = True):
    spec_info = object() if with_spec_info else None

    def is_extend(include_draft_extend_v2=False):
        return True

    forward_mode = SimpleNamespace(is_extend=is_extend, is_idle=lambda: False)
    batch = SimpleNamespace(
        forward_mode=forward_mode,
        is_extend_in_batch=False,
        has_active_mm_inputs=lambda: mm,
        spec_info=spec_info,
        seq_lens=_SeqLens([4, 6]),
        extend_lens=[3, 5],
        prefix_lens=[1, 1],
        out_cache_loc=torch.arange(8, dtype=torch.int64),
        new_seq_lens=None,
    )
    return batch


def _make_worker(capture_log):
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)

    def forward_batch_generation(batch, on_publish=None, grammar_barrier=None,
                                 pp_proxy_tensors=None, **kwargs):
        capture_log.append(kwargs.get("capture_hidden_mode"))
        batch.new_seq_lens = batch.seq_lens
        logits_output = SimpleNamespace(
            hidden_states=torch.zeros((8, 4), dtype=torch.float32),
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=torch.tensor([1, 2], dtype=torch.int64),
        )

    worker._target_worker = SimpleNamespace(forward_batch_generation=forward_batch_generation)
    worker.model_runner = SimpleNamespace(
        device="cpu", prefill_attention_backend_str="triton"
    )
    worker._tp_sync = SimpleNamespace(sync=lambda *a, **k: None)
    worker._validate_phase1_sampling_support = lambda batch: None
    worker.append_log = []
    worker._append_target_hidden_to_draft_kv_by_loc = lambda **kw: worker.append_log.append(kw)
    return worker


def _run_extend(worker, batch):
    """Drive the extend path with compute_position stubbed (backend dispatch)."""
    total = int(sum(batch.extend_lens))
    with mock.patch.object(
        dwv, "compute_position",
        lambda *a, **k: (torch.arange(total, dtype=torch.int64), None),
    ):
        return worker.forward_batch_generation(batch)


class TestDflashMmMaterialize(CustomTestCase):
    def test_mm_extend_materializes_draft_kv(self):
        """M10 fix: MM extends capture FULL and materialize like text."""
        capture_log = []
        worker = _make_worker(capture_log)
        batch = _extend_batch(mm=True)
        _run_extend(worker, batch)
        self.assertEqual(capture_log, [dwv.CaptureHiddenMode.FULL])
        self.assertEqual(len(worker.append_log), 1)
        kw = worker.append_log[0]
        self.assertEqual(kw["target_hidden"].shape[0], 8)
        self.assertEqual(int(kw["cache_loc"].numel()), 8)
        self.assertEqual(int(kw["positions"].numel()), 8)
        # spec_info restored
        self.assertIsNotNone(batch.spec_info)

    def test_mm_extend_escape_hatch_bypasses(self):
        """SGLANG_DFLASH_MM_SKIP_DRAFT_KV=1 restores the old NULL bypass."""
        capture_log = []
        worker = _make_worker(capture_log)
        batch = _extend_batch(mm=True)
        with mock.patch.object(dwv, "_MM_SKIP_DRAFT_KV", True):
            worker.forward_batch_generation(batch)
        self.assertEqual(capture_log, [dwv.CaptureHiddenMode.NULL])
        self.assertEqual(worker.append_log, [])
        self.assertIsNotNone(batch.spec_info)

    def test_text_extend_unchanged(self):
        """Text extends keep the FULL + materialize path and spec_info handling."""
        capture_log = []
        worker = _make_worker(capture_log)
        batch = _extend_batch(mm=False)
        _run_extend(worker, batch)
        self.assertEqual(capture_log, [dwv.CaptureHiddenMode.FULL])
        self.assertEqual(len(worker.append_log), 1)
        self.assertIsNotNone(batch.spec_info)


def _decode_batch(mm: bool):
    def is_extend(include_draft_extend_v2=False):
        return False

    forward_mode = SimpleNamespace(is_extend=is_extend, is_idle=lambda: False)
    original_mode = forward_mode
    spec_info = dwv.DFlashDraftInputV2.create_idle_input(device="cpu")
    batch = SimpleNamespace(
        forward_mode=forward_mode,
        _original_mode=original_mode,
        is_extend_in_batch=False,
        has_active_mm_inputs=lambda: mm,
        spec_info=spec_info,
        seq_lens=_SeqLens([4, 6]),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        input_ids=torch.tensor([9, 9], dtype=torch.int64),
        out_cache_loc=torch.arange(16, dtype=torch.int64),
        new_seq_lens=None,
    )
    return batch


class TestDflashMmDecodeFallbackScoping(CustomTestCase):
    def test_decode_fallback_restores_batch_fields(self):
        """M10: the DECODE-shaped view is scoped to the target call."""
        capture_log = []
        worker = _make_worker(capture_log)
        worker.block_size = 8
        worker.device = "cpu"
        worker.model_runner = SimpleNamespace(
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((4, 64), dtype=torch.int64)),
            prefill_attention_backend_str="triton",
        )
        worker._make_next_draft_input_decode = lambda *, bonus_tokens, new_seq_lens: object()
        batch = _decode_batch(mm=True)
        with mock.patch.object(
            dwv, "assign_extend_cache_locs_func",
            lambda **kw: torch.arange(2, dtype=torch.int64),
        ):
            result = worker.forward_batch_generation(batch)
        self.assertIsNotNone(result)
        # Batch fields restored: the DECODE view cannot leak.
        self.assertIs(batch.forward_mode, batch._original_mode)
        self.assertEqual(int(batch.input_ids[0]), 9)
        self.assertEqual(int(batch.out_cache_loc.numel()), 16)
        # spec_info restored
        self.assertIsNotNone(batch.spec_info)


if __name__ == "__main__":
    unittest.main()
