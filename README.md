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

### DFlash2 Speculative Decoding (Production)

The production systemd service (`sglang.service`) launches the DFlash2 recipe
via `scripts/rdna-serve.sh qwen38-autoround-dflash2`. Equivalent manual launch:

```bash
export SGLANG_KV_CACHE_DTYPE=bfloat16
export SGLANG_DTYPE=bfloat16
export SGLANG_PHASE_TIMING=0
export SGLANG_RDNA_CUSTOM_AR=1
export SGLANG_RDNA_NO_FUSED=1
export SGLANG_RDNA_GEMMA_TRITON=1
export SGLANG_RDNA_VLLM_VERIFY=1
export SGLANG_RDNA_AR_PATH=/path/to/rdna_ar

python -m sglang.launch_server \
  --model-path /path/to/qwen3.8-27b-mtp-fixed \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 \
  --port 8080 \
  --tp-size 2 \
  --quantization gptq \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --attention-backend triton \
  --context-length 196608 \
  --mem-fraction-static 0.90 \
  --max-running-requests 4 \
  --max-mamba-cache-size 20 \
  --cuda-graph-bs-decode 1 2 4 \
  --triton-attention-num-kv-splits 16 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm dflash \
  --speculative-draft-model-path /path/to/Qwen3.8-27B-DFlash2 \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 8 \
  --speculative-draft-window-size 2048 \
  --stream-interval 1
```

Key DFlash2 differences vs the MTP-3 recipe above:

- Draft model is the separate **Qwen3.8-27B-DFlash2** checkpoint (5-layer
  drafter, [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) on HF Hub), loaded unquantized via
  `--speculative-draft-model-quantization unquant` so it does not inherit the
  target model's W4A16 quantization config.
- `--speculative-num-draft-tokens 8` (7-token draft block + 1) with a
  `--speculative-draft-window-size 2048` sliding draft window.
- `--kv-cache-dtype bfloat16` and `--mem-fraction-static 0.90` reserve the
  dedicated draft KV budget (DFlash2 runs ~14.5 GiB total vs ~11.3 GiB for MTP-3).
- Adds `--chunked-prefill-size 2048` and the qwen3 reasoning / tool-call parsers.

Measured on this fork (2026-09-09, 2x RX 7900 XTX TP=2): math CoT TG ~166 tok/s
(GSM8K/MATH-500), 120k agentic replay mean TG ~84 tok/s, and ~97% TG retention
at 16k context depth (two-run averages).

**W4A16 draft variant (2026-09-10):**
[`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
ships the drafter pre-quantized with the same compressed-tensors W4A16 scheme as the
target — point `SGLANG_DFLASH2_PATH` at it and clear
`SGLANG_DFLASH2_QUANT` (empty) so the checkpoint's own quant config applies
instead of the `unquant` default. The drafter drops from 2.09 to **1.04
GiB/GPU** and the KV pool grows from 189,172 to **217,746 tokens** (+15.1%) at
bf16-parity accept length (3.30–3.34 solo, 3.27 @ c=4). Two loader changes
make this work: plain drafter params (`fc` + ignore-listed projections)
are dequantized from their packed checkpoint triples at load time, and the
fused context-KV path now dequantizes packed draft qkv rows via a one-hot
GPTQ GEMM (measured in §5 below).

**Greedy-mode fidelity caveat (2026-09-10):** at `temperature: 0`, block
batched verify is not bit-equivalent to sequential decode — roughly a
quarter of accept boundaries sit at a sub-0.5-logit top-2 margin, where the
M=8 verify batch's numerics can land on the other token. Committed flips are
fluent individually but re-route the chain, and long greedy generations can
drift into degraded structure (mangled tables, duplicated list items). The
accept/commit path itself is exact (per-step invariant audit, zero
violations; see `DFLASH-OUTPUT-CORRUPTION-JOURNAL.md`). MTP-3 shares the
mechanism at smaller amplitude. For quality-critical greedy workloads prefer
the MTP recipe (`qwen38-autoround-mtp`) or disable spec decode
(`SGLANG_NO_SPEC=1`); at `temperature > 0` DFlash2 is unaffected.

---

## Empirical Benchmarks

All tests conducted on 2x AMD Radeon RX 7900 XTX (TP=2) with Qwen3.8-27B-W4A16.
SGLang columns refreshed 2026-09-09 on the current merged/rebuilt build
(`llama-benchy` 0.4.0); depth-profile and 120k numbers are two-run averages,
DFlash2 math and c=4 are single runs. vLLM baseline columns are from earlier
runs and should be re-benched under the same tool version for exact deltas.
The DFlash2 columns above are the bf16 drafter; §5 re-benchmarks the DFlash2
configuration with the W4A16 drafter (2026-09-10, fused KV materialization).

### 1. Standardized Context Depth Profile (`llama-benchy`)
*Standard prompt prefill ($PP=2048$) and token generation ($TG=128$), concurrency = 1*

| Context Depth | SGLang MTP-3 (This Fork) | SGLang DFlash2 (This Fork) | vLLM MTP-3 Baseline | vLLM DFlash2 Baseline |
|:---:|:---:|:---:|:---:|:---:|
| **Depth 0** | **93.6 tok/s** | **107.9 tok/s** | 88.6 tok/s | 71.1 tok/s |
| **Depth 4,096** | **90.1 tok/s** | **96.8 tok/s** | 83.3 tok/s | 68.4 tok/s |
| **Depth 8,192** | **85.9 tok/s** | **97.7 tok/s** | 93.3 tok/s | 71.8 tok/s |
| **Depth 16,384** | **78.2 tok/s** | **105.1 tok/s** | 75.9 tok/s | 62.2 tok/s |
| **Retention (16k / 0k)** | **83.6%** | **97.4%** | 85.7% | 87.5% |

> MTP-3 vs vLLM MTP-3: +5.6 / +8.2 / −7.9 / +3.0%. DFlash2 vs vLLM DFlash2: +51.8 / +41.5 / +36.1 / +69.0%. Note the depth-trend inversion: DFlash2's windowed drafting holds or *raises* TG as context grows, while MTP-3 decays past ~4k.

### 2. Real-World 120k Agentic Session Replay (16 Progressive Turns)
*Replay across 16 discrete turns of a real agentic session (332 $\to$ 120,443 tokens) with Radix APC prefix caching*

| Metric | SGLang MTP-3 (This Fork) | SGLang DFlash2 (This Fork) | vLLM MTP-3 | vLLM DFlash2 |
|---|:---:|:---:|:---:|:---:|
| **Mean TG Speed** | **89.46 tok/s** | **83.8 tok/s** | 63.50 tok/s | 67.96 tok/s |
| **Median TG Speed** | **88.38 tok/s** | **79.8 tok/s** | 61.52 tok/s | 67.16 tok/s |
| **Jitter (CV %)** | **11.74%** | **23.0%** | 45.34% | 36.56% |
| **Worst-Case Floor** | **72.88 tok/s** | **40.6 tok/s** | 16.94 tok/s | 32.94 tok/s |

> MTP-3: **+40.9% mean** and **3.9x smoother** than vLLM MTP-3, **4.3x** its floor. DFlash2: **+23.3% mean** than vLLM DFlash2, 1.6x smoother. MTP-3 wins stability and worst-case floor; DFlash2 wins deep-context TG holding (table 1) and c=4 aggregate (table 4).

### 3. Mathematical Chain-of-Thought Reasoning (GSM8K & MATH-500)
*Greedy sampling, temperature = 0.0, max_tokens = 1024*

| Benchmark | Output Tokens | TTFT (s) | Prefill (tok/s) | Generation Speed | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| **GSM8K #1** | 48 | 0.128s | 701.6 | **97.3 tok/s** | 100% |
| **GSM8K #2** | 196 | 0.139s | 814.8 | **115.5 tok/s** | 100% |
| **MATH-500 #1** | 201 | 0.137s | 605.5 | **120.8 tok/s** | 100% |
| **MATH-500 #2** | 212 | 0.128s | 568.2 | **110.9 tok/s** | 100% |
| **Average** | — | **0.133s** | **672.5** | **113.9 tok/s** | **100%** |

**SGLang DFlash2 (same suite, single run):**

| Benchmark | Output Tokens | TTFT (s) | Prefill (tok/s) | Generation Speed | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| **GSM8K #1** | 52 | 0.131s | 689.3 | **124.8 tok/s** | ✗ (answered 96, gold 72) |
| **GSM8K #2** | 200 | 0.140s | 805.1 | **171.2 tok/s** | ✓ |
| **MATH-500 #1** | 195 | 0.137s | 606.9 | **197.7 tok/s** | ✓ |
| **MATH-500 #2** | 511 | 0.132s | 551.3 | **154.4 tok/s** | ✓ |
| **Average** | — | **0.135s** | **663.2** | **166.0 tok/s** | **75% (3/4)** |

> DFlash2 generates ~50% faster on math but slipped GSM8K #1 under greedy sampling in both runs (answered 96; gold 72) — a known trade-off of its windowed draft path on the quantized dense GEMM; MTP-3 answers all four correctly.

### 4. High Concurrency Throughput ($c=4$)
*Benchmarked via `llama-benchy` across concurrent streams ($PP=2048, TG=128$, Depth 0)*

| Serving Engine | Concurrency | Total PP Throughput | Total TG Throughput | Peak TG Throughput | Stability Notes |
|---|:---:|:---:|:---:|:---:|---|
| **SGLang MTP-3 (This Fork)** | **c = 4** | **1,357.4 tok/s** | **78.5 tok/s** | **118.0 tok/s** | **100% stable**, decode CUDA graphs + MTP-3 active |
| **SGLang DFlash2 (This Fork)** | **c = 4** | **1,275.5 tok/s** | **92.1 tok/s** | **146.0 tok/s** | **100% stable**, single sample |
| **vLLM Baseline (No Spec)** | c = 4 | 1,891.1 tok/s | 83.1 tok/s | 180.0 tok/s | Reliable baseline, but slow decode throughput |
| **vLLM DFlash2** | c = 4 | 1,693.2 tok/s | 75.9 tok/s | 188.0 tok/s | Speculative overhead reduces aggregate TG vs baseline |
| **vLLM MTP-3** | c = 4 | — | *(Crashed)* | — | Fails under batch > 1 (`hipErrorIllegalAddress`) |
| **llama.cpp (MTP)** | c = 4 | 636.3 tok/s | 50.1 tok/s | — | Bottlenecked by slot queuing (`-np 2`) |

> **Key Concurrency Takeaway**: Both SGLang spec engines stay **100% stable under 4 concurrent streams** on RDNA3 — no `hipErrorIllegalAddress`, no graph-capture failures (vLLM's native MTP-3 still crashes at batch > 1). DFlash2 leads the c = 4 field at **92.1 tok/s aggregate** (+10.8% vs the vLLM no-spec baseline, +21.3% vs vLLM DFlash2) with a **146 tok/s peak**; MTP-3 lands at 78.5 tok/s. The vLLM columns predate `llama-benchy` 0.4.0, so re-bench the baselines under 0.4.0 before quoting exact cross-engine deltas.

### 5. W4A16 Draft Model (`syvai/Qwen3.8-27B-DFlash2-W4A16`, fused KV)
*2026-09-10, same hardware and `llama-benchy` 0.4.0. Depth-profile and 120k
numbers were measured on an isolated port with no other traffic; math, c=4,
deep-context and multimodal are from the 2026-09-09 suite on the sequential
KV path (accuracy is path-independent, and fused KV only changes the draft
context-KV build).*

**Depth profile** (PP=2048/TG=128, c=1; two-run averages, 16k is a three-sample mean):

| Metric | bf16 drafter (§1) | W4A16 drafter (fused KV) |
|:---|:---:|:---:|
| **Depth 0** | 107.9 tok/s | 118.9 tok/s |
| **Depth 4,096** | 96.8 tok/s | 98.9 tok/s |
| **Depth 8,192** | 97.7 tok/s | 95.9 tok/s |
| **Depth 16,384** | 105.1 tok/s | 98.0 tok/s |
| **Retention (16k / 0k)** | 97.4% | 82.4% |

> Run-to-run TG on this box swings ±15 tok/s (back-to-back depth-0 samples read 101.7 and 136.1), so read single depths at that resolution. With fused KV disabled at the same isolated port the 16k mean drops to 88.1 tok/s — fused KV is worth ~+11% at depth and recovers most of the 16k-retention gap to the bf16 drafter.

**120k agentic replay** (two runs, fused KV): mean **88.6 tok/s**, median
**82.8 tok/s**, CV **25.3%**, worst turn **52.5 tok/s** — vs 83.8 / 79.8 /
23.0% / 40.6 for the bf16 drafter. Mean TG holds at parity or slightly better.

**Math CoT** (greedy, single run): 3/4 correct — the *same* GSM8K #1 slip as
the bf16 drafter (answered 96, gold 72), so quantizing the drafter costs no
additional accuracy. Average TG **148.9 tok/s** (the 63-token GSM8K #1 answer
skews low at 54.0 tok/s; the three longer answers run 155–200 tok/s).

**Deep-context scaling** (single run, 10k→160k TG): 78.4 / 117.3 / 90.4 /
86.1 / 60.1 / 39.0 tok/s — TG decays past ~80k, consistent with the
long-context behavior of this windowed drafter family.

**c=4 concurrency:** total TG **89.3 tok/s**, peak **121 tok/s**, PP
**1,266 tok/s**, 100% stable (bf16 drafter: 92.1 / 146.0 / 1,275.5).

**Multimodal matrix:** zero loop/format defects across single/multi-image and
interleaved agent-turn workloads.

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

---

## ROCm Linux KFD Event-Age Busy-Wait Fix (`scripts/rdna_ar/kfd_event_age_fix.c`)

When running ROCm 6.x/7.x on modern Linux kernels (KFD ABI 1.14+), you may notice **two CPU cores permanently pinned at 100% utilization** even when the server is completely idle without any incoming requests.

### Root Cause
1. **SGLang Scheduler Spin**: By default, SGLang's scheduler main loop polls active requests continuously. Adding `--sleep-on-idle` enables ZMQ socket polling (`IdleSleeper`), which drops the Python scheduler process to 0% idle CPU.
2. **ROCR Runtime Event Age Desynchronization**: Even with `--sleep-on-idle`, ROCm's runtime thread (`AsyncEventsLoop` in `libhsa-runtime64.so`) remains pegged at 100% CPU per GPU.
   - In Linux KFD ABI 1.14+, `AMDKFD_IOC_WAIT_EVENTS` uses a monotonic `event_age` counter to prevent missing event wakeups across multi-waiter threads.
   - Upstream ROCR runtime resets `event_age = 1` on the stack before invoking the ioctl. Once any GPU event has fired, the kernel's internal `ev->event_age >= 2`.
   - The kernel driver observes `ev->event_age != last_event_age` and immediately marks the event as completed (`KFD_IOC_WAIT_RESULT_COMPLETE`), returning in ~3 microseconds.
   - Userspace loops endlessly, issuing **~3.4 million ioctls per second per GPU** in pure idle.

### Resolution
We provide a zero-overhead C interposer shim (`scripts/rdna_ar/kfd_event_age_fix.c`):
- Intercepts `ioctl(AMDKFD_IOC_WAIT_EVENTS)`.
- Maintains a lock-free cache of the true kernel `last_event_age` per `event_id` in userspace.
- Supplies matching ages to AMDKFD, allowing the kernel to place the thread into true sleep (`schedule_timeout`).
- **Result**: Background idle thread CPU drops from **100% to 0.0%** per GPU without touching system packages or recompiling ROCm/PyTorch.

Build the shim:
```bash
make -C scripts/rdna_ar
```

Load via environment variable:
```bash
export LD_PRELOAD="scripts/rdna_ar/vendor/kfd_event_age_fix.so:$LD_PRELOAD"
```

