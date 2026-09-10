"""Async invariant probes — fire torch._assert_async without CPU sync.

All probes are gated on SGLANG_ENABLE_ASYNC_ASSERT (default off in prod).
When the gate is on, a violation surfaces as an assertion at the next CUDA
sync point instead of as a silent NaN cascade or illegal-address crash.
"""

import logging
import os
from typing import Optional

import torch

from sglang.srt.environ import envs

# 12.145 host-side pool guard switch (see maybe_detect_oob).
# "== 1" on purpose: bool("0") is True, so SGL_POOL_GUARD=0 used to ENABLE
# the host-sync guard it was meant to disable (REVIEW 2026-09-10 M6).
_POOL_GUARD = os.environ.get("SGL_POOL_GUARD", "0") == "1"
_WITNESS_ENABLED = os.environ.get("SGL_WITNESS_CHECK", "0") == "1"

logger = logging.getLogger(__name__)


class _AsyncNanWarner:
    """One-shot NaN monitor: device-side detection lands in pinned host
    memory without any stream sync; the host reads the (slightly stale) flag
    on a later call, warns once, and stops detecting."""

    def __init__(self):
        self._dev = None
        self._host = None
        self._warned = False

    def check(self, tensor: torch.Tensor, msg: str):
        if self._warned or not tensor.is_cuda:
            return
        if self._dev is None:
            self._dev = torch.zeros(1, dtype=torch.int32, device=tensor.device)
            self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)

        # Report a hit enqueued on an earlier step (pinned read, no sync).
        if int(self._host[0]):
            logger.warning(
                "NaN detected in %s; values were sanitized before sampling. "
                "This usually indicates numerical overflow (e.g. fp16 "
                "activations) or an upstream bug producing NaN. "
                "Logged once; further occurrences are silent.",
                msg,
            )
            self._warned = True
            return

        # Enqueue this step's detection (async, no sync).
        self._dev.add_(torch.isnan(tensor).any().to(torch.int32))
        self._host.copy_(self._dev, non_blocking=True)


_nan_warner = _AsyncNanWarner()


def maybe_warn_nan(tensor: Optional[torch.Tensor], msg: str = ""):
    """Non-fatal counterpart of maybe_detect_nan: throttled sync-free warning
    instead of crashing. Callers sanitize the tensor themselves."""
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        # The hard assert path already covers detection.
        return
    if tensor is None:
        return
    _nan_warner.check(tensor, msg)


def sanitize_nan_logits(logits: torch.Tensor, msg: str = ""):
    """Detect NaN (assert in CI, throttled warning in prod), then sanitize in
    place: NaN logits (e.g. fp16 activation overflow) are undefined behavior
    in sampling kernels and can come back as out-of-vocab token ids. +-1e30
    rather than dtype min/max because callers divide logits by temperature,
    which would overflow dtype min/max to +-Inf and softmax back to NaN."""
    maybe_detect_nan(logits, msg)
    if not envs.SGLANG_SANITIZE_NAN_LOGITS.get():
        return
    maybe_warn_nan(logits, msg)
    torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)


def maybe_assert_async(cond: torch.Tensor, msg: str = ""):
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    torch._assert_async(cond, msg)


def maybe_assert_sum(tensor: torch.Tensor, expected: int, msg: str = "") -> None:
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    torch._assert_async(tensor.sum() == expected, msg)


def maybe_detect_nan(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    # A None tensor means there is nothing to probe, e.g. hidden_states on
    # capture_hidden_mode=NULL paths (STANDALONE speculative decoding).
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


# 12.145 kernel-witness registry. Instrumented Triton kernels write
# (kernel_id, detail1, detail2) into a shared int32 tensor instead of
# trapping, so a detected violation surfaces as a named host-side error
# instead of a GPU wedge. Registered lazily by the kernel wrappers.
_WITNESS: dict = {}


def register_witness(tensor: torch.Tensor, kernel_names: dict):
    _WITNESS[tensor.device.index or 0] = (tensor, kernel_names)


def _check_kernel_witness():
    if not _WITNESS_ENABLED or not _WITNESS:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    for _dev, (tensor, names) in _WITNESS.items():
        code = int(tensor[0].item())
        if code == 0:
            continue
        d1 = int(tensor[1].item())
        d2 = int(tensor[2].item())
        name = names.get(code, f"kernel#{code}")
        head = (
            f"KERNEL WITNESS VIOLATION in {name} "
            f"(detail1={d1}, detail2={d2}, raw={tensor.tolist()})"
        )
        try:
            import traceback as _tb
            with open(os.path.expanduser("~/witness.log"), "a") as f:
                f.write(f"\n==== {head}\n")
                f.write("".join(_tb.format_stack()[-30:]) + "\n")
        except Exception:
            pass
        raise RuntimeError(head)


def maybe_detect_inf(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async Inf check — fp16 overflow surfaces as Inf before NaN."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isinf(tensor)), f"Inf detected! {msg}")


def maybe_detect_in_closed_range(
    tensor: Optional[torch.Tensor], low: float, high: float, msg: str = ""
):
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None or tensor.numel() == 0:
        return
    torch._assert_async(
        ((tensor >= low) & (tensor <= high)).all(),
        f"value outside [{low}, {high}]: {msg}",
    )


def maybe_detect_oob(indices: Optional[torch.Tensor], low: int, high: int, msg: str):
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point.

    Low/high asserted separately so the message names which failed (low =
    negative/sentinel, high = out of range).
    """
    # 12.145 witness check: instrumented kernels that detect an out-of-bounds
    # index/pointer condition on-device write (kernel_id, detail1, detail2)
    # into a small witness tensor and SKIP the offending store, so the GPU
    # never faults. Read it here — this call already syncs via the guard.
    _check_kernel_witness()
    # 12.145: host-side synchronous guard (SGL_POOL_GUARD=1). Fires BEFORE the
    # consuming kernel launches, dumping the bad index + traceback to
    # ~/pool_guard.log. Unlike torch._assert_async this cannot destabilize
    # ROCm graph capture and cannot fault the GPU.
    if _POOL_GUARD and indices is not None and indices.numel() > 0:
        import traceback as _tb
        # .item() syncs — illegal during graph capture (dummy data there anyway).
        if torch.cuda.is_current_stream_capturing():
            return
        mn = int(indices.min().item())
        mx = int(indices.max().item())
        if mn < low or mx >= high:
            head = f"POOL GUARD VIOLATION {msg}: min={mn} max={mx} allowed=[{low}, {high})"
            with open(os.path.expanduser("~/pool_guard.log"), "a") as f:
                f.write(f"\n==== {head}\n")
                f.write("".join(_tb.format_stack()[-30:]) + "\n")
            raise RuntimeError(head)
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0:
        return
    torch._assert_async(
        indices.min() >= low,
        f"index < {low} (negative / unmasked sentinel?): {msg}",
    )
    torch._assert_async(
        indices.max() < high,
        f"index >= {high} (out of range): {msg}",
    )


def maybe_detect_kernel_facing_loc(
    indices: Optional[torch.Tensor], page_size: int, blocks_per_page: int, msg: str
):
    """Async check that a write loc is in the pool's KERNEL-FACING id space.

    A kernel-facing id is `phys_page * (page_size * blocks_per_page) + offset`
    with `offset < page_size`, so its remainder modulo the page stride is
    below page_size; a VIRTUAL id satisfies that only in the first block.
    Vacuous at blocks_per_page 1. Virtual ids are in range for the OOB probe,
    so this is the only check that separates them.
    """
    if blocks_per_page <= 1 or not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0:
        return
    torch._assert_async(
        (indices % (page_size * blocks_per_page) < page_size).all(),
        f"write loc outside the kernel-facing id space (virtual ids?): {msg}",
    )


def maybe_detect_page_aligned(
    indices: Optional[torch.Tensor], page_size: int, msg: str
):
    """Async page-alignment check on slot ids."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0 or page_size <= 1:
        return
    torch._assert_async(
        (indices % page_size == 0).all(),
        f"page-misaligned indices (page_size={page_size}): {msg}",
    )
