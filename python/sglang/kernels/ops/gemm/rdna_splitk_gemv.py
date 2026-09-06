"""Split-K tall-skinny GEMV/GEMM for the vocab-parallel lm_head (HIP/RDNA3).

x [M<=8, K] * W [N, K] row-major -> out [M, N].
Grid = (N/BLOCK_N, SPLIT_K); each program loops over its K chunk in small
sub-tiles (register-friendly), writes fp32 partials; a reduce kernel sums
the slices. Targets ~700-900 GB/s on [76K, 5120] where rocBLAS MFMA
(436 GB/s at M=1) and row_dot (554 GB/s) fall short. JOURNAL 12.24/12.25.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_gemv_kernel(
    X, W, PART,  # x [M,K], W [N,K] row-major, partials [SPLIT_K, M, N] fp32
    M, N, K,
    stride_xm, stride_wn, stride_pz, stride_pm,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_base = pid_k * K_PER_SPLIT
    for kk0 in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_base + kk0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)  # [BM, BK]
        w = tl.load(
            W + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)  # [BN, BK]
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)  # [BM, BN]
    tl.store(
        PART + pid_k * stride_pz + offs_m[:, None] * stride_pm + offs_n[None, :],
        acc, mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _splitk_reduce_kernel(
    PART, OUT,  # [SPLIT_K, M, N] fp32, out [M, N]
    total, stride_pz,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for z in range(SPLIT_K):
        acc += tl.load(PART + z * stride_pz + offs, mask=mask, other=0.0)
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def splitk_gemv(x: torch.Tensor, weight: torch.Tensor, split_k: int = 8,
                block_n: int = 32, block_k: int = 64) -> torch.Tensor:
    """x [M, K] (M<=8), weight [N, K] contiguous row-major -> [M, N]."""
    M, K = x.shape
    N = weight.shape[0]
    BLOCK_M = triton.next_power_of_2(max(M, 1))
    k_per_split = triton.cdiv(K, split_k)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, block_n), split_k)
    _splitk_gemv_kernel[grid](
        x, weight, part, M, N, K,
        x.stride(0), weight.stride(0), part.stride(0), part.stride(1),
        K_PER_SPLIT=k_per_split,
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=BLOCK_M,
        num_warps=4, num_stages=2,
    )
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    total = M * N
    _splitk_reduce_kernel[(triton.cdiv(total, 1024),)](
        part, out, total, part.stride(0),
        SPLIT_K=split_k, BLOCK=1024, num_warps=4,
    )
    return out
