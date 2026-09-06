# int8 weight-only GEMV for the vocab head (RDNA3/gfx1100). JOURNAL 12.103.
# out[m,n] = scale[n] * sum_k x[m,k] * W8[n,k]; M<=4, K<=4096, N%4==0.
# Measured: M=1 609us (835GB/s), M=4 641us (794GB/s) at [124160,4096] —
# 2.1-2.2x faster than the fp16 wvSplitK (1366us, 744GB/s).
import os

_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load

        build_dir = os.path.expanduser("~/.cache/sglang/i8gemv_build")
        os.makedirs(build_dir, exist_ok=True)
        _mod = load(
            name="sglang_i8_gemv",
            sources=[os.path.join(os.path.dirname(__file__), "rdna_i8_gemv.cu")],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx1100"],
            build_directory=build_dir,
            verbose=False,
        )
    return _mod


def i8_gemv_linear(x, w8, scale):
    """x [M<=4, K] fp16, w8 [N, K] int8, scale [N] fp16 -> [M, N] fp16."""
    return _get_mod().i8_gemv(x, w8, scale)
