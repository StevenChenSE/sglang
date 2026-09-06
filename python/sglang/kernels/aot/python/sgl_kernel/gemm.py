from typing import Optional

import torch


def awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.awq_dequantize.default(qweight, scales, qzeros)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.fp8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def sgl_per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    _V2_KERNEL_SUPPORTED_GROUP_SIZES = [16, 32, 64, 128]
    if enable_v2 is None:
        enable_v2 = group_size in _V2_KERNEL_SUPPORTED_GROUP_SIZES

    if enable_v2:
        return torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
        )

    assert not fuse_silu_and_mul, "only v2 support fuse_silu_and_mul"
    assert masked_m is None, "only v2 support masked_m"
    torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit.default(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
    )


# For legacy usage
sgl_per_token_group_quant_fp8 = sgl_per_token_group_quant_8bit
sgl_per_token_group_quant_int8 = sgl_per_token_group_quant_8bit


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.sgl_per_token_quant_fp8.default(input, output_q, output_s)


def shuffle_rows(input_tensor, dst2src_map, output_tensor_shape):
    output_tensor = torch.empty(
        output_tensor_shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops.sgl_kernel.shuffle_rows.default(input_tensor, dst2src_map, output_tensor)
    return output_tensor


# GPTQ kernels
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_shuffle: bool,
    bit: int,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_gemm(
        a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit
    )


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    torch.ops.sgl_kernel.gptq_shuffle(q_weight, q_perm, bit)


def gptq_gemm_rdna3(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_qzeros: torch.Tensor,
    b_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_v2_format: bool = False,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_gemm_rdna3(
        a, b_q_weight, b_qzeros, b_scales, b_g_idx, use_v2_format
    )


def gptq_gemm_rdna3_wmma(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_qzeros: torch.Tensor,
    b_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_v2_format: bool = False,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_gemm_rdna3_wmma(
        a, b_q_weight, b_qzeros, b_scales, b_g_idx, use_v2_format
    )


def wmma_flight_set(planes: torch.Tensor) -> None:
    """12.165: install pinned-host int32 flight planes for the WMMA kernels.

    Layout: tag*2+0 = entry marks, tag*2+1 = exit marks (tags 1..7), each
    plane WMMA_FLIGHT_SLOTN = 32768 slots indexed by linear block id. Marks
    persist (never cleared) — a wedge snapshot is the last state.
    """
    torch.ops.sgl_kernel.wmma_flight_set(planes)


def moe_gptq_gemm_rdna3(
    a: torch.Tensor,
    c: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_size_m: int,
    mul_topk_weight: bool = True,
    output_topk: int = 1,
) -> None:
    torch.ops.sgl_kernel.moe_gptq_gemm_rdna3(
        a,
        c,
        b_q_weight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
    )
