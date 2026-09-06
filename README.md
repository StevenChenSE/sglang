<div align="center">

# SGLang on Dual AMD Radeon RX 7900 XTX (RDNA3 / gfx1100)

[English](README.md) | [简体中文](README.zh-CN.md)

Fork of [sgl-project/sglang](https://github.com/sgl-project/sglang) with RDNA3 / gfx1100 performance work:
Upstream main synchronization, custom RDNA3 C++/HIP AOT kernels, EAGLE MTP-3 speculative decoding, and a tuned dual-7900-XTX serving stack.

[![Upstream Synced](https://img.shields.io/badge/Upstream%20Synced-sgl--project%2Fmain-blue.svg)](https://github.com/sgl-project/sglang)
[![Architecture](https://img.shields.io/badge/Arch-AMD%20RDNA3%20%28gfx1100%29-red.svg)](https://en.wikipedia.org/wiki/RDNA_3)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## About this fork

This branch (`gfx1100-support`) carries RDNA3-specific performance work on top of upstream SGLang, fully synchronized with upstream `sglang:main` (394 commits):

- **Full Upstream Master Synchronization**: Cleanly integrated upstream's DSA indexer cooperative top-k (`csrc/elementwise/topk.hip`), `KVIndexTranslator` batch indptr translation, dense one-shot prefill routines, and speculative plan/forward stream synchronization (`plan_stream.wait_stream(fwd_stream)`).
- **RDNA3 (gfx1100) Architectural Accommodations**:
  - **Register Dequantization**: Native WMMA on RDNA3 supports BF16/FP16. In Triton decode/extend kernels, register-level dequantization (`IS_FP8_KV and _IS_RDNA`) avoids software ALU emulation penalties.
  - **Exclusion of CDNA MUBUF/Peer-IPC Custom All-Reduce**: Compiled with `-DSGL_IS_RDNA` to bypass CDNA-only memory traps, falling back safely to RDNA-optimized kernels and RCCL transport.
  - **Vector Arithmetic Operator Overloads**: Undefines `__HIP_NO_HALF_OPERATORS__` and `__HIP_NO_HALF_CONVERSIONS__` during native C++/HIP compilation for seamless clang++/hipcc vector arithmetic.
  - **Vendored Unified Verify Adapter**: Preserves the RDNA-tuned speculative verification adapter (`SGL_RDNA_VLLM_VERIFY=1`), boosting depth generation speed by +8-10% across 4k–16k contexts.
- **Hybrid GDN / Mamba State Management**: Robust support for Qwen 3.8 hybrid architectures (MHA + Mamba-2 recurrent layers) under EAGLE MTP-3 multi-step speculative decoding, with isolated intermediate SSM state buffers to prevent rollback corruption on rejected draft tokens.

---

## Preparing the model checkpoint

The target model uses a patched local checkpoint `qwen3.8-27b-mtp-fixed`, derived from
[`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ):

1. **Download the base checkpoint**:
   ```bash
   huggingface-cli download Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ --local-dir /path/to/qwen3.8-27b-mtp-fixed
   ```
2. **Issue**: The stock repo stores all 15 MTP tensors as plain **BF16** in `model_extra_tensors.safetensors`, but its `quantization_config.dynamic` declares **positive** rules `"+:.*mtp.*"` and `"+:.*mtp\.fc.*"` (4-bit, group 64). SGLang/vLLM's GPTQ loader trusts that config, tries to load MTP layers with quantized params, and fails.
3. **Patch**: In `config.json` under `quantization_config.dynamic`, replace those `"+:"` rules with a single `"-:.*mtp.*"` negative exclusion rule:
   ```json
   "dynamic": {
     "-:.*mtp.*": {
       "bits": 16,
       "group_size": 128
     }
   }
   ```
   This directs the weight loader to skip quantization on MTP layers and load the unquantized BF16 weights as-is.

---

## Hardware & Serving Environment

| Component | Specification |
|---|---|
| **GPUs** | 2x AMD Radeon RX 7900 XTX (24GB GDDR6 each, Navi 31, gfx1100, TP=2) |
| **Interconnect** | PCIe Gen 4.0 x16 (16.0 GT/s, ~31.5 GB/s bidirectional per card) |
| **Host System** | AMD Ryzen 5 9600X, DDR5 RAM |
| **ROCm Stack** | ROCm 7.14 (HIP 7.14.60850) |
| **PyTorch** | 2.11.0+git (ROCm 7.2 wheel) |
| **Model** | `Qwen3.8-27B-W4A16-AutoRound-GPTQ` (`qwen3.8-27b-mtp-fixed`) |
| **Attention Backend** | Triton Attention (`triton_backend.py`) with RDNA3 wave-aware split-K |

---

## Build & Installation

### 1. Build C++/HIP Native Ops
```bash
# Activate your ROCm Python virtual environment
source .venv-rocm/bin/activate

# Build and link the native sgl_kernel extension for gfx1100
cd sglang/python/sglang/kernels/aot
PYTORCH_ROCM_ARCH="gfx1100" python setup_rocm.py build_ext --inplace

# Copy compiled shared object into the environment site-packages
cp python/sgl_kernel/common_ops.*.so $VIRTUAL_ENV/lib/python3.12/site-packages/sgl_kernel/
```

### 2. Editable Python Installation
```bash
cd sglang/python
pip install --no-build-isolation --no-deps -e .
```

### 3. Build Standalone RDNA Custom All-Reduce Extension (Optional but Recommended)
RDNA3 GPUs lack CDNA MUBUF/peer-IPC hardware pathways. SGLang provides a standalone one-shot custom all-reduce kernel optimized for gfx1100 PCIe peer access under `scripts/rdna_ar`:

```bash
# Pre-build and seed the RDNA custom all-reduce shared object into scripts/rdna_ar/vendor
python sglang/scripts/rdna_ar/rdna_ar_ext.py
```
*Note: When `SGLANG_RDNA_CUSTOM_AR=1` is set, SGLang will auto-discover `sglang/scripts/rdna_ar` or look at `SGLANG_RDNA_AR_PATH` if specified. Pre-building ensures fast worker start without JIT compilation delays.*

---

## How to Run

Launch the production server with MTP-3 speculative decoding and Radix prefix caching:

```bash
export SGL_DTYPE=bfloat16
export SGL_PHASE_TIMING=0
export SGL_RDNA_CUSTOM_AR=1
export SGL_RDNA_NO_FUSED=1
export SGL_RDNA_GEMMA_TRITON=1
export SGL_RDNA_VLLM_VERIFY=1
# Optional: path to standalone RDNA custom allreduce extension if not in PYTHONPATH
export SGLANG_RDNA_AR_PATH=/path/to/rdna_ar

python -m sglang.launch_server \
  --model-path /path/to/qwen3.8-27b-mtp-fixed \
  --host 0.0.0.0 \
  --port 8080 \
  --served-model-name qwen3.8-27b \
  --tp-size 2 \
  --quantization gptq \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype auto \
  --attention-backend triton \
  --context-length 196608 \
  --mem-fraction-static 0.91 \
  --max-running-requests 4 \
  --max-mamba-cache-size 20 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path /path/to/qwen3.8-27b-mtp-fixed \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-bs-decode 1 2 4 \
  --triton-attention-num-kv-splits 16 \
  --sleep-on-idle
```

---

## Empirical Benchmarks

All tests conducted on 2x AMD Radeon RX 7900 XTX (TP=2) with Qwen3.8-27B-W4A16.

### 1. Standardized Context Depth Profile (`llama-benchy`)
*Standard prompt prefill ($PP=2048$) and token generation ($TG=128$), concurrency = 1*

| Context Depth | SGLang MTP-3 (This Fork) | vLLM MTP-3 Baseline | vLLM DFlash2 Baseline | Advantage vs vLLM MTP-3 |
|:---:|:---:|:---:|:---:|:---:|
| **Depth 0** | **97.1 tok/s** | 88.6 tok/s | 71.1 tok/s | **+9.6%** |
| **Depth 4,096** | **97.3 tok/s** | 83.3 tok/s | 68.4 tok/s | **+16.8%** |
| **Depth 8,192** | **91.5 tok/s** | 93.3 tok/s | 71.8 tok/s | -1.9% |
| **Depth 16,384** | **95.7 tok/s** | 75.9 tok/s | 62.2 tok/s | **+26.1%** |
| **Retention (16k / 0k)** | **98.5%** | 85.7% | 87.5% | **Rock-solid scaling** |

### 2. Real-World 120k Agentic Session Replay (16 Progressive Turns)
*Replay across 16 discrete turns of a real agentic session (332 $\to$ 120,443 tokens) with Radix APC prefix caching*

| Metric | SGLang MTP-3 (This Fork) | vLLM MTP-3 | vLLM DFlash2 | Delta / Improvement |
|---|:---:|:---:|:---:|:---:|
| **Mean TG Speed** | **87.89 tok/s** | 63.50 tok/s | 67.96 tok/s | **+38.4% faster** |
| **Median TG Speed** | **85.97 tok/s** | 61.52 tok/s | 67.16 tok/s | **+39.7% faster** |
| **Jitter (CV %)** | **12.92%** | 45.34% | 36.56% | **3.5x smoother** |
| **Worst-Case Floor** | **66.71 tok/s** | 16.94 tok/s | 32.94 tok/s | **4x higher floor** |

### 3. Mathematical Chain-of-Thought Reasoning (GSM8K & MATH-500)
*Greedy sampling, temperature = 0.0, max_tokens = 1024*

| Benchmark | Output Tokens | TTFT (s) | Prefill (tok/s) | Generation Speed | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| **GSM8K #1** | 48 | 0.169s | 533.5 | **98.9 tok/s** | 100% |
| **GSM8K #2** | 196 | 0.180s | 626.9 | **116.2 tok/s** | 100% |
| **MATH-500 #1** | 169 | 0.149s | 556.4 | **114.7 tok/s** | 100% |
| **MATH-500 #2** | 212 | 0.137s | 533.3 | **111.2 tok/s** | 100% |
| **Average** | — | **0.159s** | **562.5** | **110.3 tok/s** | **100%** |

### 4. High Concurrency Throughput ($c=4$)
*Benchmarked via `llama-benchy` across concurrent streams ($PP=2048, TG=128$, Depth 0)*

| Serving Engine | Concurrency | Total PP Throughput | Total TG Throughput | Peak TG Throughput | Stability Notes |
|---|:---:|:---:|:---:|:---:|---|
| **SGLang MTP-3 (This Fork)** | **c = 4** | **1,974.5 tok/s** | **147.5 tok/s** | **206.0 tok/s** | **100% stable**, decode CUDA graphs + MTP-3 active |
| **vLLM Baseline (No Spec)** | c = 4 | 1,891.1 tok/s | 83.1 tok/s | 180.0 tok/s | Reliable baseline, but slow decode throughput |
| **vLLM DFlash2** | c = 4 | 1,693.2 tok/s | 75.9 tok/s | 188.0 tok/s | Speculative overhead reduces aggregate TG vs baseline |
| **vLLM MTP-3** | c = 4 | — | *(Crashed)* | — | Fails under batch > 1 (`hipErrorIllegalAddress`) |
| **llama.cpp (MTP)** | c = 4 | 636.3 tok/s | 50.1 tok/s | — | Bottlenecked by slot queuing (`-np 2`) |

> **Key Concurrency Takeaway**: While earlier vLLM setups struggled with multi-stream speculative verification on ROCm (crashing or yielding lower aggregate generation speed than baseline autoregression), SGLang's wave-aligned Triton decode graphs and isolated Mamba intermediate buffers deliver **147.5 tok/s aggregate decode throughput** under 4 concurrent streams — **+77.5% faster than vLLM baseline** and **+94.4% faster than vLLM DFlash2**.

---

## Key Operational Gotchas & Guidelines

> ⚠️ **CRITICAL: NEVER USE `rocm-smi --gpureset` ON RDNA3 (gfx1100)**
> Invoking driver-level reset on Navi 31 hangs the PCIe root complex and locks the entire system, requiring a hard physical power cycle.

1. **FP8 KV Cache vs BF16**:
   - While FP8 KV halves the memory footprint, dynamic Triton indexing across deep-append prefills can trigger register spill/GPU faults on RDNA3 WMMA.
   - For 100% stable production runs, use BF16 KV cache (`--kv-cache-dtype auto`).
2. **CUDA Graphs**:
   - Pre-capture decode CUDA graphs (`--cuda-graph-bs-decode 1 2 4`). Do not enable breakable prefill capture on 24GB cards as it leaves minimal VRAM headroom at startup.
3. **Mamba Recurrent Memory Headroom**:
   - Mamba state cache allocates scratch buffers scaled to concurrency. For single-user agentic setups (concurrency $\le 4$), capping `--max-running-requests 4` and `--max-mamba-cache-size 20` frees **~2.6 GB VRAM per GPU**, allowing context expansion up to **192k** in BF16.
