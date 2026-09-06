"""Skinny-M bf16 linear (GEMV-style) kernel for RDNA3 (gfx1100).

The MTP draft layer runs bf16 F.linear at M<=8 (per draft step at bs=1).
torch/cuBLAS picks Tensile configs that run these at 20-150 GB/s on gfx1100
(measured 65-1250 us per call; the weight-read floor is 4x lower). This
memory-bound kernel assigns one CTA to a BLOCK_N-wide output stripe and
streams K, keeping all loads coalesced on the K dimension.

Splash guard: only beneficial for tiny M; the wrapper checks.
"""

import torch
import triton
import triton.language as tl

_AMD_KWARGS = {"waves_per_eu": 4} if torch.version.hip else {}


@triton.jit
def _skinny_linear_kernel(
    X,  # [M, K] (contiguous)
    W,  # [N, K] (nn.Linear layout, contiguous)
    OUT,  # [M, N]
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        # [BLOCK_K, BLOCK_N] load of W^T tiles: W is [N, K] row-major, so
        # W[n, k] with n in a stripe and k in a block is coalesced along k.
        w = tl.load(
            W + offs_n[:, None] * stride_wn + kk[None, :],
            mask=mask_n[:, None] & (kk[None, :] < K),
            other=0.0,
        )  # [BLOCK_N, BLOCK_K]
        x = tl.load(
            X + offs_m[:, None] * stride_xm + kk[None, :],
            mask=mask_m[:, None] & (kk[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        acc += tl.sum(x[:, None, :].to(tl.float32) * w[None, :, :].to(tl.float32), axis=2)

    out_ptrs = OUT + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def row_dot_kernel(
    X,  # [M, K]
    W,  # [N, K] row-major
    OUT,  # [M, N]
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Persistent grid-stride over output rows: each CTA streams W[n, :]
    # CONTIGUOUSLY (linear DRAM access) for a few rows. The original one-CTA-
    # per-row layout stops scaling at ~76K rows (vocab lm_head): per-CTA work
    # (~20B/thread) drowns in launch overhead. A capped grid (GRID_CAP CTAs,
    # each looping rows) amortizes launch while keeping the stream pattern.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    for n in range(pid, N, num_progs):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            w = tl.load(W + n * stride_wn + kk, mask=kk < K, other=0.0).to(tl.float32)
            x = tl.load(
                X + offs_m[:, None] * stride_xm + kk[None, :],
                mask=mask_m[:, None] & (kk[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x * w[None, :], axis=1)
        tl.store(OUT + offs_m * stride_om + n, acc, mask=mask_m)




@triton.jit
def row_dot_tc_kernel(
    X, W, OUT,  # x [M, K], W [N, K] row-major, out [M, N]
    M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # Tensor-core GEMM tiling for the tall-skinny vocab logits GEMM:
    # out[M, N] = x[M, K] @ W[N, K]^T. One CTA per BLOCK_N W-rows; streams
    # its W rows linearly and re-reads x per k-step (traffic ~= W traffic
    # at BLOCK_M=16). tl.dot uses MFMA units; the broadcast-sum variant
    # became L2/compute-bound at M>=4 (JOURNAL 12.27).
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(
            W + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        )  # [BN, BK] fp16/bf16
        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0,
        )  # [BM, BK]
        acc = tl.dot(w, tl.trans(x), acc)  # [BN, BM]
    # store transposed: out[m, n] = acc[n, m]
    out_ptrs = OUT + offs_m[:, None] * stride_om + offs_n[None, :]
    mask_o = (offs_m[:, None] < M) & mask_n[None, :]
    tl.store(out_ptrs, tl.trans(acc), mask=mask_o)


def row_dot_tc(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x [M<=16, K] fp16/bf16, weight [N, K] contiguous -> [M, N]."""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    row_dot_tc_kernel[(triton.cdiv(N, 32),)](
        x, weight, out, M, N, K,
        x.stride(0), weight.stride(0), out.stride(0),
        BLOCK_N=32, BLOCK_K=64, BLOCK_M=16,
        num_warps=4, num_stages=3, **_AMD_KWARGS,
    )
    return out


def skinny_linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """x: [..., K] bf16; weight: [N, K] bf16 contiguous. Returns [..., N]."""
    orig_shape = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if False and M <= 16 and K >= 2048:
        # DISABLED: tl.dot tiling measured 1643-1659us (469-473GB/s) — slower
        # than persistent row_dot. tl.dot fixes compute, not the scattered
        # 128B-per-row DRAM pattern; the full-row stream of row_dot wins.
        # Kept for reference (row_dot_tc_kernel).
        pass
    if M <= 8:
        # 12.120 sweep at the REAL lm_head shape ([M,5120]x[124160,5120]
        # bf16, cold-L2): the stripe kernel (coalesced [BN,BK] W-tiles)
        # beats persistent row_dot at M<=4 — 2009us vs 2840-3885us at M=4
        # (633 vs 448-510GB/s) and 1830us vs 2086-1920us at M=1. For
        # 4<M<=8 the broadcast-sum row_dot collapses with BLOCK_M>=8
        # (register/ALU blowup: 13.8ms at M=8); bk=8192/w=8/st=1 is the
        # least-bad row_dot config there.
        if M <= 4:
            out = torch.empty((M, N), dtype=x.dtype, device=x.device)
            if N <= 32768:
                # 12.139 sweep at the per-layer projection shapes
                # (N=3072-12288, K=3072/5120): BK=1024 with narrow
                # stripes is 1.4-2.1x over the lm_head-tuned config
                # (e.g. M=4 N=4096: 691 -> 1305GB/s). The vocab shape
                # keeps the 12.120 config below.
                BN = 8 if N <= 4096 else 16
                BK = 1024
                warps = 4
                stages = 4 if M == 1 else 2
            else:
                BN, BK, warps, stages = 32, 512, (8 if M == 1 else 4), 2
            _skinny_linear_kernel[(triton.cdiv(N, BN),)](
                x2, weight, out, M, N, K,
                x2.stride(0), weight.stride(0), out.stride(0),
                BLOCK_N=BN, BLOCK_K=BK, BLOCK_M=triton.next_power_of_2(max(M, 1)),
                num_warps=warps, num_stages=stages, **_AMD_KWARGS,
            )
        else:
            # 4<M<=16: broadcast-sum row_dot collapses at BLOCK_M>=8
            # (register/ALU blowup: 13.8ms at M=8 — JOURNAL 12.121); the
            # MFMA tl.dot tiling holds ~1.95ms (652GB/s) flat from M=8-16
            # (12.121 sweep at the real lm_head shape).
            out = torch.empty((M, N), dtype=x.dtype, device=x.device)
            row_dot_tc_kernel[(triton.cdiv(N, 32),)](
                x2, weight, out, M, N, K,
                x2.stride(0), weight.stride(0), out.stride(0),
                BLOCK_N=32, BLOCK_K=256, BLOCK_M=16,
                num_warps=4, num_stages=2, **_AMD_KWARGS,
            )
    else:
        BLOCK_N = 64
        BLOCK_K = 128
        BLOCK_M = triton.next_power_of_2(max(M, 1))
        grid = (triton.cdiv(N, BLOCK_N),)
        _skinny_linear_kernel[grid](
            x2, weight, out, M, N, K,
            x2.stride(0), weight.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            num_warps=8, num_stages=2, **_AMD_KWARGS,
        )
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], N)


@triton.jit
def row_dot_fp8_kernel(
    X,  # [M, K] bf16/fp16
    W,  # [N, K] float8_e4m3fn row-major
    SCALE,  # [N] fp32 per-row dequant scale
    OUT,  # [M, N]
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # 12.116: persistent grid-stride GEMV over an fp8-e4m3 weight with a
    # per-row scale (draft-loop lm_head: 3 x M=1 full-vocab reads per
    # scheduler step). Halves the head's DRAM traffic vs the bf16 head.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    for n in range(pid, N, num_progs):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        s = tl.load(SCALE + n).to(tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            w = tl.load(W + n * stride_wn + kk, mask=kk < K, other=0.0)
            x = tl.load(
                X + offs_m[:, None] * stride_xm + kk[None, :],
                mask=mask_m[:, None] & (kk[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x * (w.to(tl.float32) * s)[None, :], axis=1)
        tl.store(OUT + offs_m * stride_om + n, acc, mask=mask_m)


def row_dot_fp8(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    """x [M<=8, K] bf16/fp16, weight [N, K] float8_e4m3fn contiguous,
    scale [N] fp32 -> [M, N] in x.dtype."""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    row_dot_fp8_kernel[(min(N, 16384),)](
        x, weight, scale, out, M, N, K,
        x.stride(0), weight.stride(0), out.stride(0),
        BLOCK_K=4096, BLOCK_M=triton.next_power_of_2(max(M, 1)),
        num_warps=16, num_stages=2, **_AMD_KWARGS,
    )
    return out


def quantize_head_fp8(weight: torch.Tensor):
    """Per-row absmax fp8-e4m3 quantization of an lm_head weight.
    Returns (q, scale) with scale fp32 [N]."""
    w32 = weight.float()
    scale = w32.abs().amax(dim=1).clamp(min=1e-8) / 448.0
    q = (w32 / scale[:, None]).to(torch.float8_e4m3fn)
    return q, scale


@triton.jit
def fp8_stripe_kernel(
    X,  # [M, K] bf16
    W,  # [N, K] uint8 view of float8_e4m3fn, row-major
    SCALE,  # [N] fp32
    OUT,  # [M, N]
    M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # 12.116: fp8-e4m3 lm_head GEMV for the DRAFT loop (topk=1 argmax only).
    # Native fp8 loads scalarize on this Triton/gfx1100 backend (134GB/s);
    # uint8 loads + manual e4m3->bf16 bit-convert (exp bias 7->127) hit
    # 421GB/s at the real shape with argmax agreement 1.0000 (12.116).
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask_k = kk < K
        w8 = tl.load(W + offs_n[:, None] * stride_wn + kk[None, :],
                     mask=mask_n[:, None] & mask_k[None, :], other=0).to(tl.uint32)
        sgn = (w8 & 0x80) << 8
        exp = ((w8 >> 3) & 0xF) + 120
        man = (w8 & 0x7) << 4
        u16 = (sgn | (exp << 7) | man).to(tl.uint16)
        w = u16.to(tl.bfloat16, bitcast=True).to(tl.float32) \
            * tl.load(SCALE + offs_n, mask=mask_n, other=0.0)[:, None].to(tl.float32)
        x = tl.load(X + offs_m[:, None] * stride_xm + kk[None, :],
                    mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    out_ptrs = OUT + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty),
             mask=(offs_m[:, None] < M) & mask_n[None, :])


def row_dot_fp8_stripe(x: torch.Tensor, weight_u8: torch.Tensor, scale: torch.Tensor):
    """x [M<=8, K] bf16, weight_u8 [N, K] uint8 view of e4m3, scale [N] fp32."""
    M, K = x.shape
    N = weight_u8.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    fp8_stripe_kernel[(triton.cdiv(N, 32),)](
        x, weight_u8, scale, out, M, N, K,
        x.stride(0), weight_u8.stride(0), out.stride(0),
        BLOCK_N=32, BLOCK_K=512, BLOCK_M=triton.next_power_of_2(max(M, 1)),
        num_warps=8, num_stages=2, **_AMD_KWARGS,
    )
    return out
