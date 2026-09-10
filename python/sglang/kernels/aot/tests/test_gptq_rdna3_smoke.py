"""Numeric smoke test for the RDNA3 GPTQ kernels (REVIEW 2026-09-10 follow-up;
committed in-repo per round-2 review N4 so the validation is reproducible).

Uses constant nibbles so the expected dequantized weight is analytically
known regardless of the packed nibble layout: with every weight nibble = Q,
every stored zero nibble = Z, and scales = 1, the dequantized weight is
v1:  W = Q - (Z + 1)      (legacy +1 zero offset)
v2:  W = Q - Z            (raw zeros)

so for scales = 1 the kernel output must equal (a @ ones) * W for any
activation a.

Covers:
- scalar path (bf16 M < 16, fp16 M < 64) and WMMA dispatch (bf16 M >= 16,
  fp16 M >= 64, K/N % 16)
- the shim `gptq_gemm` with and without use_v2_format (H1)
- the direct `gptq_gemm_rdna3` / `gptq_gemm_rdna3_wmma` ops

Run: .venv-rocm/bin/python python/sglang/kernels/aot/tests/test_gptq_rdna3_smoke.py
"""

import sys

import pytest
import torch

Q, Z = 8, 5
V1_W = float(Q - (Z + 1))  # 2.0
V2_W = float(Q - Z)        # 3.0

K, N, GROUPS = 128, 64, 4  # gs=32; K,N % 16 == 0 so the WMMA dispatch applies


def _gfx1100_available() -> bool:
    """The scalar kernel body only has a gfx1100 code object (H4 guard), so
    the whole suite is only meaningful on that target."""
    try:
        if not torch.version.hip or not torch.cuda.is_available():
            return False
        return (
            torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1100")
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gfx1100_available(), reason="requires a gfx1100 device and the RDNA3 wheel"
)


def _const_word(nibble: int) -> int:
    """int32 word with every 4-bit nibble = nibble (wrap to signed)."""
    w = (0xF & nibble) * 0x11111111 & 0xFFFFFFFF
    return w - (1 << 32) if w >= (1 << 31) else w


def pack_constants(rows_words: int, cols: int) -> torch.Tensor:
    """int32 tensor [rows_words, cols] with every 4-bit nibble = Q."""
    return torch.full((rows_words, cols), _const_word(Q), dtype=torch.int32, device="cuda")


def pack_zeros() -> torch.Tensor:
    """int32 [groups, N/8] with every 4-bit nibble = Z."""
    return torch.full((GROUPS, N // 8), _const_word(Z), dtype=torch.int32, device="cuda")


def _run_all(check):
    from sgl_kernel import (
        gptq_gemm,
        gptq_gemm_rdna3,
        gptq_gemm_rdna3_wmma,
        gptq_shuffle,
    )

    all_ok = True
    for dtype in (torch.bfloat16, torch.float16):
        w_q = pack_constants(K // 8, N)
        w_z = pack_zeros()
        w_s = torch.ones((GROUPS, N), dtype=dtype, device="cuda")
        g_idx = torch.empty((0,), dtype=torch.int32, device="cuda")

        # Shuffle exactly like process_weights_after_loading (non-desc_act
        # passes an empty g_idx and shuffles with it).
        w_q_shuf = w_q.clone()
        gptq_shuffle(w_q_shuf, g_idx, 4)

        for M in (1, 2, 4, 8, 16, 64, 128):
            a = torch.randint(0, 2, (M, K), device="cuda").to(dtype) * 0.5
            ones = torch.ones((K, N), dtype=dtype, device="cuda")
            exp_v1 = (a @ ones) * V1_W
            exp_v2 = (a @ ones) * V2_W
            dt = str(dtype).split(".")[-1]

            all_ok &= check(f"gptq_gemm v1 {dt} M={M}",
                            gptq_gemm(a, w_q_shuf, w_z, w_s, g_idx, True, 4),
                            exp_v1)
            all_ok &= check(f"gptq_gemm v2(use_v2_format=True) {dt} M={M}",
                            gptq_gemm(a, w_q_shuf, w_z, w_s, g_idx, True, 4, True),
                            exp_v2)
            all_ok &= check(f"gptq_gemm_rdna3 v1 {dt} M={M}",
                            gptq_gemm_rdna3(a, w_q_shuf, w_z, w_s, g_idx, False),
                            exp_v1)

            wmma_floor = 16 if dtype == torch.bfloat16 else 64
            if M >= wmma_floor:
                all_ok &= check(f"gptq_gemm_rdna3_wmma v1 {dt} M={M}",
                                gptq_gemm_rdna3_wmma(a, w_q_shuf, w_z, w_s, g_idx, False),
                                exp_v1)
    return all_ok


def test_gptq_rdna3_numeric_smoke():
    torch.cuda.init()

    failures = []

    def check(tag, out, expected):
        maxdiff = (out - expected).abs().max().item()
        ok = torch.allclose(out, expected, atol=2e-2, rtol=2e-2)
        if not ok:
            failures.append(f"{tag}: maxdiff={maxdiff:.4g}")
        return ok

    assert _run_all(check), "numeric smoke failures:\n" + "\n".join(failures)


if __name__ == "__main__":
    torch.cuda.init()

    def check(tag, out, expected):
        maxdiff = (out - expected).abs().max().item()
        ok = torch.allclose(out, expected, atol=2e-2, rtol=2e-2)
        print(f"  {'OK ' if ok else 'FAIL'} {tag:52s} maxdiff={maxdiff:.4g}")
        return ok

    ok = _gfx1100_available() and _run_all(check)
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
