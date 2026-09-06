<div align="center">

# SGLang (RDNA3 / gfx1100 Optimized Fork)

**High-Performance LLM & Speculative Serving for AMD RDNA3 (Navi 31 / RX 7900 XTX / gfx1100)**

[English](#english) | [中文](#chinese)

[![Upstream Synced](https://img.shields.io/badge/Upstream%20Synced-sgl--project%2Fmain-blue.svg)](https://github.com/sgl-project/sglang)
[![Architecture](https://img.shields.io/badge/Arch-AMD%20RDNA3%20%28gfx1100%29-red.svg)](https://en.wikipedia.org/wiki/RDNA_3)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

<a name="english"></a>
## English

### About This Fork

This repository is an optimized fork of [sgl-project/sglang](https://github.com/sgl-project/sglang) tailored specifically for AMD RDNA3 consumer hardware (**Navi 31 / Radeon RX 7900 XTX / gfx1100**), fully synced with latest upstream `sglang:main`.

#### Key Highlights & Custom Enhancements
- **Upstream Synchronization**: Merged with latest upstream `main` (394 commits ahead), adopting upstream DSA indexer cooperative top-k (`topk.hip`), `KVIndexTranslator`, dense one-shot prefill, and speculative stream synchronization.
- **RDNA3 (gfx1100) Architectural Safeguards**:
  - **Register Dequantization**: Native WMMA on RDNA3 supports BF16/FP16; register-level dequantization (`IS_FP8_KV and _IS_RDNA`) avoids software ALU emulation penalties.
  - **RCCL Fallback / Non-CDNA IPC**: Excludes CDNA-specific MUBUF/peer-IPC custom all-reduce traps via `-DSGL_IS_RDNA` in favor of RDNA-tuned implementations and RCCL transport.
  - **Vector Arithmetic Overloads**: Undefines `__HIP_NO_HALF_OPERATORS__` and `__HIP_NO_HALF_CONVERSIONS__` during native C++/HIP compilation for seamless clang++/hipcc vector arithmetic.
  - **Vendored Unified Verify Adapter**: Preserved RDNA-tuned speculative verification adapter (`SGL_RDNA_VLLM_VERIFY=1`), boosting depth generation speed by +8-10% at 4k–16k contexts.
- **Hybrid GDN / Mamba State Management**: Stable support for Qwen 3.8 hybrid architectures (MHA + Mamba-2 recurrent layers) with EAGLE MTP-3 multi-step speculative decoding.

---

### Hardware & Environment Specifications

| Component | Specification |
|---|---|
| **GPUs** | 2x AMD Radeon RX 7900 XTX (24GB GDDR6 each, Navi 31, gfx1100, TP=2) |
| **Interconnect** | PCIe Gen 4.0 x16 (16.0 GT/s, ~31.5 GB/s bidirectional per card) |
| **Host System** | AMD Ryzen 5 9600X, DDR5 RAM |
| **ROCm Stack** | ROCm 7.14 (HIP 7.14.60850) |
| **PyTorch** | 2.11.0+git (ROCm 7.2 wheel) |
| **Model** | `Qwen3.8-27B-W4A16-AutoRound-GPTQ` (MTP fixed checkpoint) |
| **Attention Backend** | Triton Attention (`triton_backend.py`) with RDNA3 wave-aware split-K |

---

### Build & Installation

#### 1. Build C++/HIP Native Ops
```bash
# Activate your ROCm Python virtual environment
source .venv-rocm/bin/activate

# Build and link the native sgl_kernel extension for gfx1100
cd sglang/python/sglang/kernels/aot
PYTORCH_ROCM_ARCH="gfx1100" python setup_rocm.py build_ext --inplace

# Copy compiled shared object into the environment site-packages
cp python/sgl_kernel/common_ops.*.so $VIRTUAL_ENV/lib/python3.12/site-packages/sgl_kernel/
```

#### 2. Editable Python Installation
```bash
cd sglang/python
pip install --no-build-isolation --no-deps -e .
```

---

### How to Run

Launch the production server with MTP-3 speculative decoding and Radix prefix caching:

```bash
export SGL_DTYPE=bfloat16
export SGL_PHASE_TIMING=0
export SGL_RDNA_CUSTOM_AR=1
export SGL_RDNA_NO_FUSED=1
export SGL_RDNA_GEMMA_TRITON=1
export SGL_RDNA_VLLM_VERIFY=1

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
  --context-length 131072 \
  --mem-fraction-static 0.88 \
  --max-running-requests 20 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path /path/to/qwen3.8-27b-mtp-fixed \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-bs-decode 1 2 4 8 16 20 \
  --triton-attention-num-kv-splits 16
```

---

### Empirical Benchmarks

All tests conducted on 2x AMD Radeon RX 7900 XTX (TP=2) with Qwen3.8-27B-W4A16.

#### 1. Standardized Context Depth Profile (`llama-benchy`)
*Standard prompt prefill ($PP=2048$) and token generation ($TG=128$), concurrency = 1*

| Context Depth | SGLang MTP-3 (This Fork) | vLLM MTP-3 Baseline | vLLM DFlash2 Baseline | Advantage vs vLLM MTP-3 |
|:---:|:---:|:---:|:---:|:---:|
| **Depth 0** | **97.1 tok/s** | 88.6 tok/s | 71.1 tok/s | **+9.6%** |
| **Depth 4,096** | **97.3 tok/s** | 83.3 tok/s | 68.4 tok/s | **+16.8%** |
| **Depth 8,192** | **91.5 tok/s** | 93.3 tok/s | 71.8 tok/s | -1.9% |
| **Depth 16,384** | **95.7 tok/s** | 75.9 tok/s | 62.2 tok/s | **+26.1%** |
| **Retention (16k / 0k)** | **98.5%** | 85.7% | 87.5% | **Rock-solid scaling** |

#### 2. Real-World 120k Agentic Session Replay (16 Progressive Turns)
*Replay across 16 discrete turns of a real agentic session (332 $\to$ 120,443 tokens) with Radix APC prefix caching*

| Metric | SGLang MTP-3 (This Fork) | vLLM MTP-3 | vLLM DFlash2 | Delta / Improvement |
|---|:---:|:---:|:---:|:---:|
| **Mean TG Speed** | **87.89 tok/s** | 63.50 tok/s | 67.96 tok/s | **+38.4% faster** |
| **Median TG Speed** | **85.97 tok/s** | 61.52 tok/s | 67.16 tok/s | **+39.7% faster** |
| **Jitter (CV %)** | **12.92%** | 45.34% | 36.56% | **3.5x smoother** |
| **Worst-Case Floor** | **66.71 tok/s** | 16.94 tok/s | 32.94 tok/s | **4x higher floor** |

#### 3. Mathematical Chain-of-Thought Reasoning (GSM8K & MATH-500)
*Greedy sampling, temperature = 0.0, max_tokens = 1024*

| Benchmark | Output Tokens | TTFT (s) | Prefill (tok/s) | Generation Speed | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| **GSM8K #1** | 48 | 0.169s | 533.5 | **98.9 tok/s** | 100% |
| **GSM8K #2** | 196 | 0.180s | 626.9 | **116.2 tok/s** | 100% |
| **MATH-500 #1** | 169 | 0.149s | 556.4 | **114.7 tok/s** | 100% |
| **MATH-500 #2** | 212 | 0.137s | 533.3 | **111.2 tok/s** | 100% |
| **Average** | — | **0.159s** | **562.5** | **110.3 tok/s** | **100%** |

---

### Key Operational Gotchas & Guidelines

> ⚠️ **CRITICAL: NEVER USE `rocm-smi --gpureset` ON RDNA3 (gfx1100)**
> Invoking driver-level reset on Navi 31 hangs the PCIe root complex and locks the entire system, requiring a hard physical power cycle.

1. **FP8 KV Cache vs BF16**:
   - While FP8 KV halves the memory footprint, dynamic Triton indexing across deep-append prefills can trigger register spill/GPU faults on RDNA3 WMMA.
   - For 100% stable production runs, use BF16 KV cache (`--kv-cache-dtype auto`).
2. **CUDA Graphs**:
   - Pre-capture decode CUDA graphs (`--cuda-graph-bs-decode 1 2 4 ...`). Do not enable breakable prefill capture on 24GB cards as it leaves minimal VRAM headroom at startup.
3. **Mamba Recurrent Memory Headroom**:
   - Mamba state cache allocates scratch buffers scaled to concurrency. For single-user agentic setups (concurrency $\le 4$), capping `--max-running-requests 4` and `--max-mamba-cache-size 20` frees **~2.6 GB VRAM per GPU**, allowing context expansion up to **192k** in BF16.

---

<a name="chinese"></a>
## 中文

### 关于本 Fork

本仓库是 [sgl-project/sglang](https://github.com/sgl-project/sglang) 针对 AMD RDNA3 消费级显卡（**Navi 31 / Radeon RX 7900 XTX / gfx1100**）的性能优化分支，已完全合入最新上游官方 `sglang:main`。

#### 核心优化与架构适配
- **全面同步上游 Master**：合入最新 394 个 commits，包含 DSA 索引协同 Top-K 算子（`topk.hip`）、`KVIndexTranslator` 批量索引映射、Dense One-shot Prefill 以及推测验证流时序对齐。
- **针对 RDNA3 (gfx1100) 硬件特性的深层适配**：
  - **寄存器级反量化**：RDNA3 WMMA 单元原生支持 BF16/FP16。在 Triton 注意力内核中通过寄存器转换反量化（`IS_FP8_KV and _IS_RDNA`），避免软仿真惩罚。
  - **规避 CDNA 陷阱 / 自定义 All-Reduce**：通过 `-DSGL_IS_RDNA` 宏禁用 CDNA 特有的 MUBUF / 跨卡 Peer-IPC 内存陷阱，回退到经过 RDNA 调优的高效实现与标准 RCCL 通信。
  - **矢量运算符重载支持**：编译 HIP 原生算子时取消定义 `__HIP_NO_HALF_OPERATORS__` 与 `__HIP_NO_HALF_CONVERSIONS__`，确保 Clang++/HIPCC 矢量算术重载无缝通过。
  - **集成 Unified Verify 验证适配器**：保留 RDNA 统一推测验证核（`SGL_RDNA_VLLM_VERIFY=1`），在 4k–16k 深度上下文生成中取得 +8-10% 的持续速度提升。
- **混合 GDN / Mamba 架构推测解码**：深度支持 Qwen 3.8 混合架构（全注意力 + Mamba-2 循环层），在 EAGLE MTP-3 多步推测下保持零回滚污染与高接受率。

---

### 硬件与运行环境

| 组件 | 配置参数 |
|---|---|
| **显卡** | 2x AMD Radeon RX 7900 XTX (每张 24GB GDDR6, Navi 31, gfx1100, TP=2) |
| **总线连接** | PCIe Gen 4.0 x16 (16.0 GT/s，单卡双向带宽约 31.5 GB/s) |
| **CPU / 内存** | AMD Ryzen 5 9600X, DDR5 高速内存 |
| **ROCm 驱动栈** | ROCm 7.14 (HIP 7.14.60850) |
| **PyTorch** | 2.11.0+git (ROCm 7.2 官方构建 wheel) |
| **模型** | `Qwen3.8-27B-W4A16-AutoRound-GPTQ` (MTP 修复版权重) |
| **注意力后端** | Triton Attention (`triton_backend.py`)，开启 RDNA3 Wave32 切分 |

---

### 构建与安装

#### 1. 编译 C++/HIP 原生算子库
```bash
# 激活 ROCm Python 虚拟环境
source .venv-rocm/bin/activate

# 针对 gfx1100 架构编译 sgl_kernel 原生扩展
cd sglang/python/sglang/kernels/aot
PYTORCH_ROCM_ARCH="gfx1100" python setup_rocm.py build_ext --inplace

# 将编译完成的 .so 动态链接库同步至虚拟环境 site-packages
cp python/sgl_kernel/common_ops.*.so $VIRTUAL_ENV/lib/python3.12/site-packages/sgl_kernel/
```

#### 2. 安装可编辑 Python 包
```bash
cd sglang/python
pip install --no-build-isolation --no-deps -e .
```

---

### 生产启动命令

通过环境变量开启 RDNA3 调优加速，启动支持 EAGLE MTP-3 推测解码的 OpenAI 兼容推理服务：

```bash
export SGL_DTYPE=bfloat16
export SGL_PHASE_TIMING=0
export SGL_RDNA_CUSTOM_AR=1
export SGL_RDNA_NO_FUSED=1
export SGL_RDNA_GEMMA_TRITON=1
export SGL_RDNA_VLLM_VERIFY=1

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
  --context-length 131072 \
  --mem-fraction-static 0.88 \
  --max-running-requests 20 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path /path/to/qwen3.8-27b-mtp-fixed \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-bs-decode 1 2 4 8 16 20 \
  --triton-attention-num-kv-splits 16
```

---

### 实测性能基准对比

全部基准均在双卡 RX 7900 XTX (TP=2) 实机测试，运行 Qwen3.8-27B-W4A16 模型。

#### 1. 标准化上下文深度衰减测试 (`llama-benchy`)
*标准 Prompt Prefill ($PP=2048$) 与 Token Generation ($TG=128$), 并发数 = 1*

| 上下文深度 | SGLang MTP-3 (本分支) | vLLM MTP-3 基线 | vLLM DFlash2 基线 | 对比 vLLM MTP-3 优势 |
|:---:|:---:|:---:|:---:|:---:|
| **Depth 0** | **97.1 tok/s** | 88.6 tok/s | 71.1 tok/s | **+9.6%** |
| **Depth 4,096** | **97.3 tok/s** | 83.3 tok/s | 68.4 tok/s | **+16.8%** |
| **Depth 8,192** | **91.5 tok/s** | 93.3 tok/s | 71.8 tok/s | -1.9% |
| **Depth 16,384** | **95.7 tok/s** | 75.9 tok/s | 62.2 tok/s | **+26.1%** |
| **速度留存率 (16k / 0k)** | **98.5%** | 85.7% | 87.5% | **长文本衰减极低** |

#### 2. 真实 120k Agent 多轮会话回放（16 轮离散交互）
*采样自真实 120k 长文本 Agent 对话（332 $\to$ 120,443 tokens），启用 Radix 前缀缓存*

| 评估指标 | SGLang MTP-3 (本分支) | vLLM MTP-3 | vLLM DFlash2 | 提升幅度 |
|---|:---:|:---:|:---:|:---:|
| **平均生成速度 (Mean TG)** | **87.89 tok/s** | 63.50 tok/s | 67.96 tok/s | **提速 +38.4%** |
| **中位数速度 (Median TG)** | **85.97 tok/s** | 61.52 tok/s | 67.16 tok/s | **提速 +39.7%** |
| **抖动率 (CV Jitter %)** | **12.92%** | 45.34% | 36.56% | **平稳度提升 3.5 倍** |
| **最差轮次保底速度** | **66.71 tok/s** | 16.94 tok/s | 32.94 tok/s | **最低速度提升 4 倍** |

#### 3. 数学思维链推理测试 (GSM8K & MATH-500)
*Greedy 贪婪采样，temperature = 0.0，max_tokens = 1024*

| 数据集用例 | 生成 Token 数量 | 首字延迟 (TTFT) | Prefill 速度 | 生成速度 (TG) | 准确率 |
|---|:---:|:---:|:---:|:---:|:---:|
| **GSM8K #1** | 48 | 0.169s | 533.5 tok/s | **98.9 tok/s** | 100% 正确 |
| **GSM8K #2** | 196 | 0.180s | 626.9 tok/s | **116.2 tok/s** | 100% 正确 |
| **MATH-500 #1** | 169 | 0.149s | 556.4 tok/s | **114.7 tok/s** | 100% 正确 |
| **MATH-500 #2** | 212 | 0.137s | 533.3 tok/s | **111.2 tok/s** | 100% 正确 |
| **综合平均** | — | **0.159s** | **562.5 tok/s** | **110.3 tok/s** | **100% 正确** |

---

### 运维注意事项与避坑指南

> ⚠️ **高危操作提示：切勿在 RDNA3 (gfx1100) 环境执行 `rocm-smi --gpureset`**
> 在 Navi 31 架构上对 GPU 发送驱动级重置指令会导致 PCIe 根端口锁死、全机卡死，只能通过机箱物理冷重启恢复。

1. **FP8 与 BF16 KV Cache 选型**：
   - 尽管 FP8 能节省一半 KV 显存，但在极长上下文的动态追加 Prefill 中容易诱发 Triton 动态索引计算导致的内核故障。
   - 生产级稳定运行请坚持使用 BF16 KV 缓存（`--kv-cache-dtype auto`）。
2. **CUDA 图调度**：
   - 推荐仅对 Decode 阶段录制 CUDA 图（`--cuda-graph-bs-decode 1 2 4 ...`）。在 24GB 显存显卡上切勿启用 Prefill 阶段图捕获，以防启动显存不足。
3. **Mamba 循环状态显存优化**：
   - Mamba 状态缓存按并发数预分配。单人 Agent 使用场景下，设置 `--max-running-requests 4` 与 `--max-mamba-cache-size 20` 可释放单卡约 **2.6 GB 显存**，直接将 BF16 窗口推至 **192k (196,608 tokens)**。
