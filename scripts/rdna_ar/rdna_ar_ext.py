# Standalone RDNA3 custom all-reduce extension.
#
# Compiles csrc/allreduce/custom_all_reduce.hip (already hipified, generic HIP
# C++) into a small standalone torch extension via torch.utils.cpp_extension,
# bypassing the full sgl_kernel AOT rebuild. Exposes the same function surface
# as sgl_kernel.allreduce (HIP flavor).
#
# Enable in sglang with SGLANG_RDNA_CUSTOM_AR=1 (see custom_all_reduce_ops.py).
#
# Self-patching: the AOT header uses NVIDIA PTX inline asm for the flag
# load/store (invalid under AMD clang: "invalid input constraint 'l'"), and a
# hipified CU_POINTER_ATTRIBUTE_* constant. We copy the sources into the build
# dir and rewrite them there, so the AOT tree itself is never modified.

import os
import shutil
import sys
import sysconfig

import torch
from torch.utils import cpp_extension

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
_AOT_AR_DIR = os.path.join(_REPO_ROOT, "python/sglang/kernels/aot/csrc/allreduce")
_AOT_INCLUDE = os.path.join(_REPO_ROOT, "python/sglang/kernels/aot/include")

_BUILD_KEY = "sgl_rdna_ar_v18"

_RELEASE_ST = r'''static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
#ifdef USE_MUSA
  volatile_store((uint32_t)flag, (uint32_t*)flag_addr);
#else
  __hip_atomic_store(flag_addr, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}'''

_ACQUIRE_LD = r'''  FlagType flag;
  flag = __hip_atomic_load(flag_addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  __hip_atomic_store(flag_addr, flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  flag = __hip_atomic_load(flag_addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return flag;
}'''


def _patch_header(text: str) -> str:
    """Replace PTX-asm flag ops with HIP system-scope atomics (RDNA3-safe)."""
    out = text

    # st_flag_release: drop the __CUDA_ARCH__ PTX branch
    out = out.replace(
        '''#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif''',
        '''#else
  __hip_atomic_store(flag_addr, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif''',
    )

    # ld_flag_acquire + st_flag_volatile + ld_flag_volatile: replace PTX loads/stores
    out = out.replace(
        '''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
  return flag;''',
        '''  flag = __hip_atomic_load(flag_addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  return flag;''',
    )

    out = out.replace(
        '''  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));''',
        '''  __hip_atomic_store(flag_addr, flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);''',
    )

    out = out.replace(
        '''  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;''',
        '''  flag = __hip_atomic_load(flag_addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return flag;''',
    )

    # hipify leftover constant name
    out = out.replace(
        "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR",
        "HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR",
    )

    # 12.115 live-context verifier: append an end-of-kernel check to
    # cross_device_reduce_1stage (ws=2 focus). Thread 0 of block 0
    # independently recomputes out[0..3] from both ranks' registered buffers
    # and counts mismatches into the wrapper-defined device globals (visible
    # because the wrapper defines them before including this header). Runs on
    # EVERY launch, including inside captured graphs — zero sync overhead.
    anchor = """  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {"""
    verifier = """  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
  if (sglang_g_ar_dbg_enabled) {
    if constexpr (!std::is_same<T, float>::value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      atomicAdd(&sglang_g_ar_dbg_calls, 1u);
      const T* dbg_a = (const T*)_dp->ptrs[0];
      const T* dbg_b = (const T*)_dp->ptrs[1];
      bool dbg_bad = false;
      // 12.116: front + stride + tail probes — a partial copy_sys_kernel
      // (stale buffer tail) is invisible to a first-4-only check.
      int dbg_idx[7] = {0, 1, 2, 3, size >> 2, size >> 1, size - 1};
      for (int t = 0; t < 7; t++) {
        int i = dbg_idx[t];
        float dbg_sum = upcast_s(dbg_a[i]) + upcast_s(dbg_b[i]);
        T dbg_exp = downcast_s<T>(dbg_sum);
        volatile const T* dbg_rv = (volatile const T*)result;
        if (upcast_s(dbg_rv[i]) != upcast_s(dbg_exp)) dbg_bad = true;
      }
      if (dbg_bad) atomicAdd(&sglang_g_ar_dbg_mismatch, 1u);
    }
    }
  }
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {"""
    assert anchor in out, "1stage kernel tail anchor not found"
    out = out.replace(anchor, verifier)

    assert "asm volatile" not in out or '"l"' not in out, "PTX asm with 'l' constraint still present"
    return out


def _load():
    # Vendor the built .so next to the ext and load it directly when present;
    # seed it after every successful build so future runs survive /tmp wipes.
    vendor_dir = os.path.join(_THIS_DIR, "vendor")
    vendor_so = os.path.join(vendor_dir, f"{_BUILD_KEY}.so")
    if os.path.exists(vendor_so):
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location(_BUILD_KEY, vendor_so)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            print(
                f"[rdna_ar] vendored .so load failed ({e!r}); falling back "
                "to JIT build"
            )

    build_dir = os.path.join("/tmp", _BUILD_KEY)
    src_dir = os.path.join(build_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    hip_src = os.path.join(_AOT_AR_DIR, "custom_all_reduce.hip")
    cuh_src = os.path.join(_AOT_AR_DIR, "custom_all_reduce_hip.cuh")

    dst_cu = os.path.join(src_dir, "custom_all_reduce.cu")
    dst_cuh = os.path.join(src_dir, "custom_all_reduce_hip.cuh")

    wrapper_src = os.path.join(_THIS_DIR, "rdna_custom_all_reduce.cu")

    needs_update = (
        not os.path.exists(dst_cu)
        or os.path.getmtime(dst_cu) < os.path.getmtime(wrapper_src)
        or not os.path.exists(dst_cuh)
        or os.path.getmtime(dst_cuh) < os.path.getmtime(cuh_src)
    )
    if needs_update:
        shutil.copyfile(wrapper_src, dst_cu)
        with open(cuh_src) as f:
            header = f.read()
        with open(dst_cuh, "w") as f:
            f.write(_patch_header(header))

    extra_includes = [src_dir, _AOT_INCLUDE]
    py_inc = sysconfig.get_path("include")
    if py_inc and os.path.isdir(py_inc):
        extra_includes.append(py_inc)
    for p in [os.environ.get("PYDEV_INCLUDE_PATH"), "/usr/include/python3.12", "/usr/include/x86_64-linux-gnu"]:
        if p and os.path.isdir(p) and p not in extra_includes:
            extra_includes.append(p)

    mod = cpp_extension.load(
        name=_BUILD_KEY,
        sources=[dst_cu],
        extra_cuda_cflags=["-O3", "-DHIP_FP8_TYPE_E4M3"],
        extra_cflags=["-O3"],
        extra_include_paths=extra_includes,
        verbose=False,
        build_directory=build_dir,
    )

    # Seed the vendor cache so future boots survive a /tmp wipe.
    try:
        os.makedirs(vendor_dir, exist_ok=True)
        built_so = os.path.join(build_dir, f"{_BUILD_KEY}.so")
        if os.path.exists(built_so):
            shutil.copyfile(built_so, vendor_so)
    except Exception as e:
        print(f"[rdna_ar] vendor seeding failed: {e!r}")
    return mod


_mod = _load()

init_custom_ar = _mod.init_custom_ar
all_reduce_reg = _mod.all_reduce_reg
all_reduce_unreg = _mod.all_reduce_unreg
register_buffer = _mod.register_buffer
get_graph_buffer_ipc_meta = _mod.get_graph_buffer_ipc_meta
register_graph_buffers = _mod.register_graph_buffers
allocate_meta_buffer = _mod.allocate_meta_buffer
get_meta_buffer_ipc_handle = _mod.get_meta_buffer_ipc_handle
meta_size = _mod.meta_size
dispose = _mod.dispose
debug_ptrs = _mod.debug_ptrs
debug_kernel_read = _mod.debug_kernel_read
copy_into_reg_buffer = _mod.copy_into_reg_buffer
allocate_reg_buffer = _mod.allocate_reg_buffer
allocate_reg_buffer_guarded = _mod.allocate_reg_buffer_guarded
ar_dbg_set = _mod.ar_dbg_set
ar_dbg_pop = _mod.ar_dbg_pop

if __name__ == "__main__":
    print(f"[rdna_ar] Successfully compiled and loaded {_BUILD_KEY} extension.")
