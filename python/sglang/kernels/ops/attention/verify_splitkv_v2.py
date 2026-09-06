"""v2 of _verify_prefix_stage1: split-D variant for HEAD_DIM=256 on gfx1100.

Identical signature/semantics to verify_splitkv._verify_prefix_stage1, but
the 256-wide D dimension is processed in two 128-wide halves inside the
loop: the K/V register tiles halve, enabling BLOCK_N=64 + num_stages=3 at
the same register budget (the stock kernel is register-pinned at BN=32 —
JOURNAL 12.39/12.46/12.53). Offline-testable via sweep_worker_v2.py.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _verify_prefix_stage1_v2(
    Q, K_Buffer, V_Buffer,
    sm_scale, k_scale, v_scale,
    qo_indptr, kv_indptr, kv_indices,
    Att_Out, Att_Lse,
    stride_qbs, stride_qh,
    stride_buf_kbs, stride_buf_kh,
    stride_buf_vbs, stride_buf_vh,
    stride_ob, stride_oh, stride_os, stride_ol,
    stride_lb, stride_lh, stride_ls,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
):
    HALF_D: tl.constexpr = HEAD_DIM // 2
    HALF_DV: tl.constexpr = V_HEAD_DIM // 2

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    offs_d1 = tl.arange(0, HALF_D)              # first D half
    offs_d2 = HALF_D + tl.arange(0, HALF_D)     # second D half
    offs_dv1 = tl.arange(0, HALF_DV)
    offs_dv2 = HALF_DV + tl.arange(0, HALF_DV)
    offs_l = tl.arange(0, L_EXT)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([L_EXT], dtype=tl.float32)
    acc1 = tl.zeros([L_EXT, HALF_DV], dtype=tl.float32)
    acc2 = tl.zeros([L_EXT, HALF_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # q halves: [L_EXT, HALF_D] each (coalesced row-major loads)
        q1 = tl.load(
            Q + (cur_q_start + offs_l)[:, None] * stride_qbs
            + cur_head * stride_qh + offs_d1[None, :],
            mask=mask_l[:, None], other=0.0,
        )
        q2 = tl.load(
            Q + (cur_q_start + offs_l)[:, None] * stride_qbs
            + cur_head * stride_qh + offs_d2[None, :],
            mask=mask_l[:, None], other=0.0,
        )
        q1 = q1.to(K_Buffer.dtype.element_ty)
        q2 = q2.to(K_Buffer.dtype.element_ty)

        base_k1 = cur_kv_head * stride_buf_kh + offs_d1[:, None]
        base_k2 = cur_kv_head * stride_buf_kh + offs_d2[:, None]
        base_v1 = cur_kv_head * stride_buf_vh + offs_dv1[None, :]
        base_v2 = cur_kv_head * stride_buf_vh + offs_dv2[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask, other=0,
            )
            k1 = tl.load(
                K_Buffer + kv_loc[None, :] * stride_buf_kbs + base_k1,
                mask=n_mask[None, :], other=0.0,
            )
            k2 = tl.load(
                K_Buffer + kv_loc[None, :] * stride_buf_kbs + base_k2,
                mask=n_mask[None, :], other=0.0,
            )
            qk = tl.dot(q1, k1)
            qk = tl.dot(q2, k2, qk) * (sm_scale * k_scale)
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            v1 = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs + base_v1,
                mask=n_mask[:, None], other=0.0,
            )
            v2 = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs + base_v2,
                mask=n_mask[:, None], other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc1 = acc1 * re_scale[:, None] + tl.dot(p.to(v1.dtype), v1)
            acc2 = acc2 * re_scale[:, None] + tl.dot(p.to(v2.dtype), v2)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        acc1 *= v_scale
        acc2 *= v_scale

        o_base = (
            cur_batch * stride_ob + cur_head * stride_oh
            + split_kv_id * stride_os + offs_l[:, None] * stride_ol
        )
        tl.store(
            Att_Out + o_base + offs_dv1[None, :], acc1 / e_sum[:, None],
            mask=mask_l[:, None],
        )
        tl.store(
            Att_Out + o_base + offs_dv2[None, :], acc2 / e_sum[:, None],
            mask=mask_l[:, None],
        )
        offs_lse = (
            cur_batch * stride_lb + cur_head * stride_lh
            + split_kv_id * stride_ls + offs_l
        )
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_l)
    else:
        offs_lse = (
            cur_batch * stride_lb + cur_head * stride_lh
            + split_kv_id * stride_ls + offs_l
        )
        tl.store(
            Att_Lse + offs_lse,
            tl.zeros([L_EXT], tl.float32) - float("inf"),
            mask=mask_l,
        )
