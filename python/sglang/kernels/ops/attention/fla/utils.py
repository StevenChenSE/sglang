# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# -*- coding: utf-8 -*-

import contextlib
import functools
import inspect
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
import triton
from packaging import version

from sglang.srt.utils.common import torch_release

logger = logging.getLogger(__name__)

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"


SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)

autotune_cache_kwargs = (
    {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
)


@lru_cache(maxsize=1)
def check_environments():
    """
    Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.
    """
    # Check Operating System
    if sys.platform == "win32":
        logger.warning(
            "Detected Windows operating system. Triton does not have an official Windows release, "
            "thus FLA will not be adapted for Windows, and any potential errors will not be fixed. "
            "Please consider using a Linux environment for compatibility."
        )

    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.2.0")

    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.2.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton."
        )

    # Check Python version
    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience."
        )

    return None


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            import warnings

            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))


# 12.149 (r13 follow-up): the identity key is only sound if the argument
# tensors are never mutated in place between calls. SGL_FLA_CACHE_VERIFY
# closes that assumption with data: on an identity hit, re-fingerprint the
# tensor arguments and compare with the fingerprints captured at cache time.
#   unset / 0     : off (zero overhead, original behaviour)
#   1             : verify; on mismatch log the caller to ~/fla_cache_verify.log
#                   and RECOMPUTE (correctness restored, evidence kept)
#   raise         : verify; on mismatch raise (named, with stack)
# Fingerprinting syncs tiny metadata tensors (cu_seqlens) — the cached helpers
# already device→host them internally, so no new sync class is introduced.
def _fla_cache_verify_mode() -> str:
    from sglang.srt.utils.flight_flags import get_flag

    return get_flag("FLA_CACHE_VERIFY", os.getenv("SGL_FLA_CACHE_VERIFY", "0"))

_FLA_CACHE_VERIFY = None  # resolved lazily (file flags + env fallback)


def _fla_fingerprint(t: torch.Tensor):
    flat = t.detach().reshape(-1)
    return (tuple(flat.shape), str(flat.dtype), flat.tolist())


def _fla_verify_mismatch(fn, args, kwargs, entry_fps) -> bool:
    import traceback

    mismatched = []
    for idx, (a, fp) in enumerate(zip(args, entry_fps)):
        if fp is None or not isinstance(a, torch.Tensor):
            continue
        if _fla_fingerprint(a) != fp:
            mismatched.append(idx)
    if not mismatched:
        return False
    head = (
        f"FLA CACHE STALENESS: {fn.__name__} identity hit with mutated tensor "
        f"arg(s) {mismatched} (cached result was stale; recomputing)"
    )
    try:
        with open(os.path.expanduser("~/fla_cache_verify.log"), "a") as f:
            f.write(f"\n==== {head}\n")
            f.write("".join(traceback.format_stack()[-25:]) + "\n")
    except Exception:
        pass
    if _fla_cache_verify_mode() == "raise":
        raise RuntimeError(head)
    return True


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.
    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.
    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry[0], entry[1], entry[2]
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    if _fla_cache_verify_mode() != "0":
                        entry_fps = entry[3]
                        if _fla_verify_mismatch(fn, args, kwargs, entry_fps):
                            break  # stale: fall through to recompute
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result, entry[3] if len(entry) > 3 else None)]
                    )
                    return last_result

        result = fn(*args, **kwargs)

        entry_fps = None
        if _fla_cache_verify_mode() != "0":
            entry_fps = [
                _fla_fingerprint(a) if isinstance(a, torch.Tensor) else None
                for a in args
            ]
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result, entry_fps))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version

            require_version(version, hint)
            return fn(
                ctx,
                *(
                    i if not isinstance(i, torch.Tensor) else i.contiguous()
                    for i in args
                ),
                **{
                    k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                    for k, v in kwargs.items()
                },
            )

        return wrapper

    return decorator


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)

    return wrapper


def _cpu_device_warning():
    import warnings

    warnings.warn(
        ("Triton is not supported on current platform, roll back to CPU."), stacklevel=1
    )


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)[
            "multiprocessor_count"
        ]
    except BaseException:
        _cpu_device_warning()
        return -1


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return "cpu"


@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    if device == "cuda":
        return "nvidia"
    elif device == "hip":
        return "amd"
    elif device == "xpu":
        return "intel"
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != "hip" else "cuda"
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()

is_amd = device_platform == "amd"
is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8
is_gather_supported = hasattr(triton.language, "gather")


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)[
                "max_shared_mem"
            ]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


if torch_release >= (2, 4):
    device = "cuda" if device == "cpu" else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)

else:
    assert (
        device == "cuda"
    ), "Only cuda device is supported for PyTorch version < 2.4.0."
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


device_platform = get_available_device()


# 12.145 kernel-witness: instrumented GDN/conv kernels write (kernel_id,
# detail1, detail2) here and skip the offending store when their index
# arithmetic detects an out-of-bounds condition, so the violation surfaces
# as a named host error at the next sync instead of a wild GPU write.
_WITNESS_TENSOR = None
WITNESS_KERNEL_NAMES = {
    1: "chunk_gated_delta_rule_fwd_h",
    2: "chunk_gated_delta_rule_fwd_kkt_solve",
    3: "recompute_w_u_fwd",
    4: "chunk_fwd_o",
    5: "chunk_local_cumsum_scalar",
    6: "chunk_local_cumsum_vector",
    7: "causal_conv1d_fwd",
    8: "fused_slot_clear (COW clear)",
    9: "fused_slot_copy (COW copy)",
    10: "triton_mrope_fused (position gather)",
    11: "extend_attention_fwd (_fwd_kernel)",
}


def get_witness(device):
    global _WITNESS_TENSOR
    if _WITNESS_TENSOR is None:
        _WITNESS_TENSOR = torch.zeros(16, dtype=torch.int32, device=device)
        try:
            from sglang.srt.utils.async_probe import register_witness

            register_witness(_WITNESS_TENSOR, WITNESS_KERNEL_NAMES)
        except Exception:
            pass
    return _WITNESS_TENSOR
