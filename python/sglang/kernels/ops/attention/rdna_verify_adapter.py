# SPDX-License-Identifier: Apache-2.0
"""ROCm/gfx1100 spec-verify adapter for the vendored vLLM unified kernel.

Adoption of ``rdna_unified_verify.py`` (vLLM triton_unified_attention_diffkv)
for sglang's target-verify path. See JOURNAL.md 12.134-12.137.

Contract with the sglang triton backend (target-verify mode):
  * ``k_buffer`` / ``v_buffer``: pool tensors [pool_size, H_KV, D] fp8
    (page_size = 1). Draft KV is already written to the pool before
    attention runs (backend saves KV first).
  * ``kv_indices`` / ``kv_indptr``: per-sequence UNIFIED walk
    (prefix + draft slots), int64 / int32.
  * ``qo_indptr``: constant stride num_draft_tokens; CUDA-graph static.
  * fp8 KV scales must be 1.0 (upstream kernel mis-handles non-unity
    scales — JOURNAL 12.135 defect 3); the adapter hard-asserts.

CUDA-graph safety: every buffer is persistent and pre-allocated; the
only per-call work is a metadata fill kernel with static launch shape
plus scalar host ints derived from the capture batch size.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.rdna_unified_verify import unified_attention_diffkv

SEGMENTS = 16          # NUM_PAR_SOFTMAX_SEGMENTS upstream
SEQ_THRESHOLD_3D = 512  # their 3D eligibility: total tokens must be <= this


@triton.jit
def _fill_block_table_kernel(
    kv_indices_ptr,  # int64 [kv_indptr[-1]]
    kv_indptr_ptr,   # int32 [bs]
    seq_lens_ptr,    # int32 [bs]
    table_ptr,       # int32 [max_bs, max_pages]
    stride_row: tl.int32,
    MAX_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(kv_indptr_ptr + b)
    seq_len = tl.load(seq_lens_ptr + b)
    # runtime bound: only walk the pages this sequence can touch
    n_iters = tl.cdiv(seq_len, BLOCK)
    for j0 in range(0, n_iters * BLOCK, BLOCK):
        offs = j0 + tl.arange(0, BLOCK)
        page = tl.load(
            kv_indices_ptr + start + offs, mask=offs < seq_len, other=0
        ).to(tl.int32)
        tl.store(
            table_ptr + b.to(tl.int64) * stride_row + offs,
            page,
            mask=offs < MAX_PAGES,
        )


class RdnaVerifyBuffers:
    """Persistent, graph-safe scratch for the adapter. One instance per
    attention backend; sizes cover the worst capture shape."""

    def __init__(
        self,
        device,
        max_bs: int,
        max_pages: int,
        num_draft_tokens: int,
        h_q: int,
        head_dim: int,
        block_q: int,
        segments: int = 16,
        head_dim_v: int | None = None,
    ):
        self.max_bs = max_bs
        self.max_pages = max_pages
        self.num_draft_tokens = num_draft_tokens
        self.block_q = block_q
        self.segments = segments
        self.blocks_per_seq = (num_draft_tokens + block_q - 1) // block_q

        self.block_table = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.kv_indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        # static per capture: qo_indptr and q_block_offsets
        self.qo_indptr = (
            torch.arange(
                0,
                (max_bs + 1) * num_draft_tokens,
                step=num_draft_tokens,
                dtype=torch.int32,
                device=device,
            )
        )
        self.q_block_offsets = (
            torch.arange(
                0,
                (max_bs + 1) * self.blocks_per_seq,
                step=self.blocks_per_seq,
                dtype=torch.int32,
                device=device,
            )
        )
        # segm partials are TOKEN-ROW indexed (JOURNAL 12.136)
        max_tokens = max_bs * num_draft_tokens
        # The unified kernel linearly indexes segm_output with
        # HEAD_SIZE_V_PADDED strides (next_pow2 of the V head dim). Allocating
        # the last dim with the same padded value keeps the layout correct and
        # in-bounds even for DiffKV models where next_pow2(v) > qk head dim
        # (REVIEW 2026-09-10 M5). For qk == v this equals head_dim.
        segm_head_dim = triton.next_power_of_2(
            head_dim_v if head_dim_v is not None else head_dim
        )
        self.segm_output = torch.empty(
            (max_tokens, h_q, segments, segm_head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.segm_max = torch.empty(
            (max_tokens, h_q, segments), dtype=torch.float32, device=device
        )
        self.segm_expsum = torch.empty(
            (max_tokens, h_q, segments), dtype=torch.float32, device=device
        )

    def fill(
        self,
        bs: int,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Per-call metadata fill. All tensor shapes are static (padded
        buffers + views of length bs); only values change."""
        self.seq_lens[:bs] = seq_lens.to(torch.int32)
        _fill_block_table_kernel[(bs,)](
            kv_indices,
            kv_indptr,
            self.seq_lens,
            self.block_table,
            self.block_table.stride(0),
            MAX_PAGES=self.max_pages,
            BLOCK=1024,
            num_warps=4,
        )


@triton.jit
def _fill_unified_extend_kernel(
    kv_indices_ptr,  # int (flat, ragged) prefix-only walk
    prefix_indptr_ptr,  # int32 [bs+1]
    ocl_ptr,  # int (flat) out_cache_loc, req-major, ntok per req
    table_ptr,  # int32 [max_bs, max_pages]
    seq_lens_ptr,  # int32 [bs] FULL lens (prefix + extend)
    stride_row: tl.int32,
    ntok: tl.int32,
    MAX_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """DRAFT_EXTEND_V2 unified walk (JOURNAL 12.141): row b = prefix
    walk followed by the req's extend tokens (out_cache_loc), total
    = seq_lens[b]."""
    b = tl.program_id(0)
    p_start = tl.load(prefix_indptr_ptr + b)
    p_end = tl.load(prefix_indptr_ptr + b + 1)
    kv_len = p_end - p_start
    full_len = tl.load(seq_lens_ptr + b)
    ext_len = full_len - kv_len
    base = b.to(tl.int64) * stride_row
    n1 = tl.cdiv(kv_len, BLOCK)
    for j0 in range(0, n1 * BLOCK, BLOCK):
        offs = j0 + tl.arange(0, BLOCK)
        v = tl.load(kv_indices_ptr + p_start + offs, mask=offs < kv_len, other=0)
        tl.store(table_ptr + base + offs, v.to(tl.int32), mask=offs < MAX_PAGES)
    o_base = b * ntok
    n2 = tl.cdiv(ext_len, BLOCK)
    for j0 in range(0, n2 * BLOCK, BLOCK):
        offs = j0 + tl.arange(0, BLOCK)
        v = tl.load(ocl_ptr + o_base + offs, mask=offs < ext_len, other=0)
        # mask MUST bound by ext_len: an offs < MAX_PAGES mask here lets
        # zero stores spill past the row end and race the next row's
        # writes (JOURNAL 12.141 defect).
        tl.store(
            table_ptr + base + kv_len + offs,
            v.to(tl.int32),
            mask=(offs < ext_len) & ((kv_len + offs) < MAX_PAGES),
        )


def rdna_fill_extend(
    bufs: RdnaVerifyBuffers,
    kv_indices: torch.Tensor,
    prefix_indptr: torch.Tensor,
    out_cache_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    bs: int,
    num_tokens_per_req: int,
    full_kv_indptr: torch.Tensor,
) -> None:
    """Fill bufs.table with the unified prefix+extend walk for
    DRAFT_EXTEND_V2 and write the FULL indptr into bufs.kv_indptr.
    Caller then invokes rdna_verify_fwd(skip_fill=True)."""
    if bs > bufs.max_bs:
        raise ValueError(f"draft-extend bs {bs} exceeds adapter sizing")
    bufs.seq_lens[:bs].copy_(seq_lens[:bs], non_blocking=True)
    bufs.kv_indptr[: bs + 1].copy_(full_kv_indptr[: bs + 1], non_blocking=True)
    _fill_unified_extend_kernel[(bs,)](
        kv_indices,
        prefix_indptr,
        out_cache_loc,
        bufs.block_table,
        bufs.seq_lens,
        bufs.block_table.stride(0),
        num_tokens_per_req,
        MAX_PAGES=bufs.max_pages,
        BLOCK=1024,
        num_warps=4,
    )


def rdna_verify_fwd(
    q: torch.Tensor,            # [bs * draft, H_Q, D] bf16
    o: torch.Tensor,            # [bs * draft, H_Q, D] bf16 (written)
    k_buffer: torch.Tensor,     # [pool, H_KV, D_QK] fp8
    v_buffer: torch.Tensor,     # [pool, H_KV, D_V] fp8
    kv_indptr: torch.Tensor,    # int32 [bs+1] (unified)
    kv_indices: torch.Tensor,   # int64 [total] (unified)
    seq_lens: torch.Tensor,     # int32/int64 [bs] (context + draft)
    sm_scale: float,
    bufs: RdnaVerifyBuffers,
    bs: int,
    num_draft_tokens: int,
    softcap: float = 0.0,
    k_scale=None,
    v_scale=None,
    logit_cap: float = 0.0,
    skip_fill: bool = False,
) -> None:
    """Write verify attention into ``o``. Raises on unsupported config
    so the caller can fall back to the in-tree kernel. With
    skip_fill=True the caller has already filled bufs.table /
    bufs.seq_lens / bufs.kv_indptr (draft-extend unified walk) and
    kv_indices is only a placeholder."""
    if k_scale is not None or v_scale is not None:
        raise ValueError("rdna_verify_fwd requires unity fp8 KV scales")
    if logit_cap not in (0.0, 0.1, 1.0) and logit_cap > 0.0:
        # upstream supports softcap; be conservative until validated
        raise ValueError("rdna_verify_fwd requires logit_cap == 0")
    if bs * num_draft_tokens > SEQ_THRESHOLD_3D:
        raise ValueError("batch too large for the 3D split-KV path")
    if bs > bufs.max_bs or num_draft_tokens != bufs.num_draft_tokens:
        raise ValueError("batch exceeds adapter buffer sizing")

    h_q = q.shape[1]
    h_kv = k_buffer.shape[1]
    d_qk = q.shape[2]
    d_v = v_buffer.shape[2]
    BQ = 16 // (h_q // h_kv)
    total_q_blocks = bufs.blocks_per_seq * bs

    # Per-call metadata fill (static shapes, graph-safe).
    if not skip_fill:
        bufs.fill(bs, kv_indices, kv_indptr, seq_lens)
    else:
        bufs.seq_lens[:bs].copy_(seq_lens[:bs], non_blocking=True)
        kv_indptr = bufs.kv_indptr[: bs + 1]

    # page_size=1 views of the pool (no copy).
    k_view = k_buffer.unsqueeze(1)
    v_view = v_buffer.unsqueeze(1)

    unified_attention_diffkv(
        q,
        k_view,
        v_view,
        o,
        bufs.qo_indptr[: bs + 1],
        bufs.seq_lens[:bs],
        sm_scale,
        True,               # causal (spec-verify chain semantics)
        (-1, -1),           # no sliding window
        bufs.block_table,
        softcap,
        max_seqlen_q=num_draft_tokens,
        seq_threshold_3D=SEQ_THRESHOLD_3D,
        num_par_softmax_segments=bufs.segments,
        softmax_segm_output=bufs.segm_output,
        softmax_segm_max=bufs.segm_max,
        softmax_segm_expsum=bufs.segm_expsum,
        q_block_offsets=bufs.q_block_offsets,
        total_q_blocks=total_q_blocks,
    )
