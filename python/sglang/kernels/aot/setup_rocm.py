# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


import sysconfig

operator_namespace = "sgl_kernel"
py_include = Path(sys.prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}"
if not (py_include / "Python.h").exists():
    py_include = root.parent.parent.parent.parent / "pydev/usr/include" / f"python{sys.version_info.major}.{sys.version_info.minor}"

include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
    py_include,
]

sources = [
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/allreduce/deterministic_all_reduce.hip",
    "csrc/allreduce/quick_all_reduce.cu",
    "csrc/common_extension_rocm.cc",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/deepseek_v4_topk.cu",
    "csrc/elementwise/dsv4_norm_rope.cu",
    # Native HIP implementation of the same three ops exposed by topk.cu.
    "csrc/elementwise/topk.hip",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/elementwise/pos_enc.cu",
    "csrc/quantization/gguf/gguf_kernel.cu",
    "csrc/gemm/gptq/gptq_kernel.cu",
    "csrc/gemm/wv_skinny_gemms.cu",
    "csrc/gemm/gptq/q_gemm_rdna3.cu",
    "csrc/gemm/gptq/q_gemm_rdna3_wmma.cu",
    "csrc/gemm/gptq/moe_q_gemm_rdna3.cu",
]

cxx_flags = ["-O3"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")

# RDNA consumer / APU targets (wave32)
RDNA_TARGETS = {"gfx1100", "gfx1151", "gfx1201"}
is_rdna = amdgpu_target in RDNA_TARGETS

if amdgpu_target not in ["gfx942", "gfx950", "gfx1250", "gfx1100", "gfx1151", "gfx1201"]:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        "Expected 'gfx942', 'gfx950', 'gfx1250', 'gfx1100', 'gfx1151', or 'gfx1201'."
    )
    sys.exit(1)

# On RDNA the CDNA-only all-reduce collectives are not built (no MUBUF / peer-IPC
# fast paths there; multi-GPU falls back to RCCL), so drop their sources.
if is_rdna:
    sources = [s for s in sources if not s.startswith("csrc/allreduce/")]

fp8_macro = (
    "-DHIP_FP8_TYPE_FNUZ" if amdgpu_target == "gfx942" else "-DHIP_FP8_TYPE_E4M3"
)  # gfx950 and gfx1250 use E4M3

# Dynamic shared-memory budget for the TopK kernels.
# - gfx942 (MI300/MI325): LDS is typically 64KB per workgroup -> keep dynamic smem <= ~48KB
#   (leaves room for static shared allocations in the kernel).
# - gfx95x (MI350) and gfx1250: LDS is larger. Large dynamic budget wastes LDS
#   and pins occupancy to 1 block/CU. Keep it small (40KB) for better occupancy.
# - RDNA (gfx1100/gfx1151/gfx1201): 64KB LDS per workgroup -> same 48KB budget as gfx942.
topk_dynamic_smem_bytes = 48 * 1024 if amdgpu_target in ("gfx942", *RDNA_TARGETS) else 40 * 1024

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--amdgpu-target={amdgpu_target}",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    fp8_macro,
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
    # Match the vLLM fork's HIP profile: torch's COMMON_HIPCC_FLAGS defines
    # __HIP_NO_HALF_OPERATORS__/__HIP_NO_HALF_CONVERSIONS__, which disables the
    # half/bf16 operator overloads and measurably degrades the gptq RDNA3
    # kernels' codegen. vLLM undefines both (-U after -D wins); do the same.
    # Flags appended here come AFTER torch's -D on the hipcc command line.
    "-U__HIP_NO_HALF_OPERATORS__",
    "-U__HIP_NO_HALF_CONVERSIONS__",
]

# On RDNA the CDNA-only all-reduce collectives are not built; guard their
# registration (common_extension_rocm.cc) and declarations (sgl_kernel_ops.h).
# The flag must reach BOTH compilers: hipcc for the .hip/.cu sources (headers)
# and the host C++ compiler for common_extension_rocm.cc (a .cc file), otherwise
# the registration is compiled in and links against the excluded symbols.
if is_rdna:
    hipcc_flags.append("-DSGL_IS_RDNA")
    cxx_flags.append("-DSGL_IS_RDNA")

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": hipcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
