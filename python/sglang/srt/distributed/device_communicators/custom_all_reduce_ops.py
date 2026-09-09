# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/_custom_ops.py
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch

from sglang.srt.utils import is_cuda, is_hip, is_musa

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()

IS_CUSTOM_AR_AVAILABLE = _is_cuda or _is_hip or _is_musa
IS_QUICK_AR_AVAILABLE = _is_hip

try:
    import sgl_kernel.allreduce as _custom_ar
except ImportError as e:
    if _is_cuda or _is_hip:
        logger.warning("Failed to import from custom_ar with %r", e)
    IS_CUSTOM_AR_AVAILABLE = False
    IS_QUICK_AR_AVAILABLE = False

# RDNA3 fallback: sgl_kernel's allreduce ops are not compiled for RDNA targets
# (setup_rocm.py drops csrc/allreduce from the source list). With
# SGLANG_RDNA_CUSTOM_AR=1, load the standalone JIT extension
# (scripts/rdna_ar/rdna_ar_ext.py) which exposes the same HIP custom-AR API on
# top of the AOT custom_all_reduce_hip.cuh, patched for gfx1100:
#  - flag load/store via HIP system-scope atomics (NVIDIA PTX asm is invalid
#    under AMD clang)
#  - peer data buffers must be dedicated hipExtMallocWithFlags(uncached)
#    allocations: PCIe peer kernel reads only observe HBM, not the writer's
#    L2 dirty lines, so torch caching-allocator segments read back zeros.
_RDNA_CUSTOM_AR = os.environ.get("SGLANG_RDNA_CUSTOM_AR", "0") == "1"
if _RDNA_CUSTOM_AR:
    try:
        _rdna_ar_path = os.environ.get("SGLANG_RDNA_AR_PATH")
        if not _rdna_ar_path or not os.path.isdir(_rdna_ar_path):
            _in_tree = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../../../scripts/rdna_ar",
                )
            )
            if os.path.isdir(_in_tree):
                _rdna_ar_path = _in_tree
        if _rdna_ar_path and os.path.isdir(_rdna_ar_path):
            sys.path.insert(0, _rdna_ar_path)
        import rdna_ar_ext as _custom_ar  # type: ignore[no-redef]

        IS_CUSTOM_AR_AVAILABLE = True
        # rdna_ar_ext only exports the custom-AR surface; the quick-AR
        # (qr_*) wrappers below would bind to missing symbols and raise
        # AttributeError if ever invoked. gfx1100 is outside
        # qr_rocm_arch_available() anyway, so keep the flag honest.
        IS_QUICK_AR_AVAILABLE = False
        logger.warning(
            "SGLANG_RDNA_CUSTOM_AR=1: using standalone RDNA custom allreduce "
            "extension (rdna_ar_ext); sgl_kernel AR ops are overridden"
        )
        # 12.116 layout probe: allocate MB of uncached memory EARLY (same
        # allocator/heap the custom-AR buffers use) without any of the AR
        # init side effects. With --disable-custom-all-reduce this isolates
        # "uncached allocation shifts the memory layout" from "peer access /
        # IPC registration".
        _dummy_mb = int(os.environ.get("SGL_RDNA_DUMMY_UNCACHED_MB", "0"))
        if _dummy_mb:
            _RDNA_DUMMY_UNCACHED = _custom_ar.allocate_reg_buffer(
                _dummy_mb * 1024 * 1024
            )
            logger.warning(
                "SGL_RDNA_DUMMY_UNCACHED_MB=%d allocated (layout probe)",
                _dummy_mb,
            )
        # 12.116 stage-2 probe: allocate the meta buffer (uncached) and
        # rank_data WITHOUT exporting any IPC handle and WITHOUT calling
        # init_custom_ar. Corrupt => the allocations themselves poison the
        # run; healthy => the poison needs the IPC/ctor path.
        _probe_stage = int(os.environ.get("SGL_RDNA_PROBE_STAGE", "0"))
        if _probe_stage >= 2:
            _RDNA_PROBE_META = _custom_ar.allocate_meta_buffer(
                _custom_ar.meta_size() + 16 * 1024 * 1024
            )
            _RDNA_PROBE_RANK_DATA = torch.empty(
                8 * 1024 * 1024, dtype=torch.uint8,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            logger.warning(
                "SGL_RDNA_PROBE_STAGE=2: meta+rank_data allocated "
                "(no IPC handles, no init)"
            )
        if _probe_stage >= 3:
            # 12.116 stage-3: LOCAL hipIpcGetMemHandle exports on the
            # uncached buffers (no peer open, no init, no execution).
            _RDNA_PROBE_META_HANDLE = _custom_ar.get_meta_buffer_ipc_handle(
                _RDNA_PROBE_META
            )
            _RDNA_PROBE_REG = _custom_ar.allocate_reg_buffer(16 * 1024 * 1024)
            logger.warning("SGL_RDNA_PROBE_STAGE=3: local IPC handles exported")
        if _probe_stage >= 4:
            # 12.116 stage-4: + init_custom_ar (the C++ ctor: signals table,
            # rank_data setup). Peer signals are null under
            # SGL_RDNA_SKIP_META_IPC, so nothing peer-related is touched and
            # the AR is never executed here. Corrupt => the ctor is the poison.
            _dummy_h = [b"\0" * 64, b"\0" * 64]
            _RDNA_PROBE_FA = _custom_ar.init_custom_ar(
                _RDNA_PROBE_META, _RDNA_PROBE_RANK_DATA, _dummy_h, [0, 0],
                torch.cuda.current_device(), True,
            )
            logger.warning("SGL_RDNA_PROBE_STAGE=4: init_custom_ar ctor ran")
    except Exception as e:
        logger.warning(
            "SGLANG_RDNA_CUSTOM_AR=1 but failed to load rdna_ar_ext: %r", e
        )
        IS_CUSTOM_AR_AVAILABLE = False

# region IS_CUSTOM_AR_AVAILABLE

if not IS_CUSTOM_AR_AVAILABLE:
    pass

elif _is_cuda or _is_musa:
    # CUDA custom allreduce

    def init_custom_ar(
        ipc_tensors: List[torch.Tensor],
        rank_data: torch.Tensor,
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return _custom_ar.init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        _custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)

    def dispose(fa: int) -> None:
        _custom_ar.dispose(fa)

    def meta_size() -> int:
        return _custom_ar.meta_size()

    def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
        return _custom_ar.register_buffer(fa, ipc_tensors)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return _custom_ar.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        _custom_ar.register_graph_buffers(fa, handles, offsets)

elif _is_hip:
    # ROCM custom allreduce

    def init_custom_ar(
        meta: torch.Tensor,
        rank_data: torch.Tensor,
        handles: List[str],
        offsets: List[int],
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return _custom_ar.init_custom_ar(
            meta, rank_data, handles, offsets, rank, full_nvlink
        )

    def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
        _custom_ar.all_reduce_reg(fa, inp, out)

    def all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        _custom_ar.all_reduce_unreg(fa, inp, reg_buffer, out)

    def deterministic_all_reduce_reg(
        fa: int, inp: torch.Tensor, out: torch.Tensor
    ) -> None:
        _custom_ar.deterministic_all_reduce_reg(fa, inp, out)

    def deterministic_all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        _custom_ar.deterministic_all_reduce_unreg(fa, inp, reg_buffer, out)

    def dispose(fa: int) -> None:
        _custom_ar.dispose(fa)

    def meta_size() -> int:
        return _custom_ar.meta_size()

    def register_buffer(
        fa: int, t: torch.Tensor, handles: List[str], offsets: List[int]
    ) -> None:
        return _custom_ar.register_buffer(fa, t, handles, offsets)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[torch.Tensor, List[int]]:
        return _custom_ar.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[str], offsets: List[List[int]]
    ) -> None:
        _custom_ar.register_graph_buffers(fa, handles, offsets)

    def allocate_meta_buffer(size: int) -> torch.Tensor:
        return _custom_ar.allocate_meta_buffer(size)

    def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor:
        return _custom_ar.get_meta_buffer_ipc_handle(inp)

    def allocate_reg_buffer(size: int) -> torch.Tensor:
        """Dedicated uncached allocation for the shared copy-in buffer.

        Only available with the standalone RDNA extension; PCIe peer kernel
        reads observe HBM only, so the buffer must avoid L2 write-back.
        """
        return _custom_ar.allocate_reg_buffer(size)

    # JOURNAL 12.115: bind the fused AR+RMSNorm wrapper only when the
    # extension actually provides a working kernel. A wrapper over a missing
    # (throwing) extension symbol turns every fused attempt into a caught
    # exception mid-graph-capture; gating on hasattr keeps the fallback
    # (custom AR + separate RMSNorm) exception-free.
    if hasattr(_custom_ar, "fused_allreduce_rmsnorm"):

        def fused_allreduce_rmsnorm(
            fa: int,
            inp: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            reg_buffer: torch.Tensor,
            out_normed: torch.Tensor,
            out_residual: torch.Tensor,
        ) -> None:
            """Fused all-reduce + residual-add + RMSNorm (standalone RDNA ext, ws=2)."""
            _custom_ar.fused_allreduce_rmsnorm(
                fa, inp, residual, weight, eps, reg_buffer, out_normed, out_residual
            )

    # 12.115: in-kernel AR verification controls (rdna_ar v14 debug build).
    if hasattr(_custom_ar, "ar_dbg_set"):

        def ar_dbg_set(enabled: int) -> None:
            """Enable/disable the end-of-kernel AR check (every launch)."""
            _custom_ar.ar_dbg_set(enabled)

    if hasattr(_custom_ar, "ar_dbg_pop"):

        def ar_dbg_pop() -> Tuple[int, int]:
            """Read+reset (mismatches, calls) counters from the AR kernel."""
            return _custom_ar.ar_dbg_pop()


# endregion

# region IS_QUICK_AR_AVAILABLE

if not IS_QUICK_AR_AVAILABLE:
    pass

elif _is_hip:
    # ROCM custom quick allreduce

    def init_custom_qr(
        rank: int, world_size: int, qr_max_size: Optional[int] = None
    ) -> int:
        return _custom_ar.init_custom_qr(world_size, rank, qr_max_size)

    def qr_get_handle(fa: int) -> torch.Tensor:
        return _custom_ar.qr_get_handle(fa)

    def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
        _custom_ar.qr_open_handles(fa, handles)

    def qr_all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        quant_level: int,
        cast_bf2half: bool,
    ) -> None:
        _custom_ar.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)

    def qr_destroy(fa: int) -> None:
        _custom_ar.qr_destroy(fa)

    def qr_max_size() -> int:
        return _custom_ar.qr_max_size()


# endregion
