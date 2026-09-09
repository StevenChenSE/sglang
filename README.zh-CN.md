<div align="center">

# SGLang 双卡 AMD Radeon RX 7900 XTX (RDNA3 / gfx1100) 深度优化分支

[English](README.md) | [简体中文](README.zh-CN.md)

基于 [sgl-project/sglang](https://github.com/sgl-project/sglang) 的 RDNA3 / gfx1100 性能优化分支：
深度同步官方上游 Master、保留 RDNA3 原生 C++/HIP AOT 算子、支持 EAGLE MTP-3 多步推测解码与双卡 RX 7900 XTX (TP=2) 生产级服务栈。

[![Upstream Synced](https://img.shields.io/badge/Upstream%20Synced-sgl--project%2Fmain-blue.svg)](https://github.com/sgl-project/sglang)
[![Architecture](https://img.shields.io/badge/Arch-AMD%20RDNA3%20%28gfx1100%29-red.svg)](https://en.wikipedia.org/wiki/RDNA_3)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 关于本 Fork

本分支（`gfx1100-support`）在 SGLang 基础上进行了深度的 RDNA3 硬件架构适配，并与上游官方最新 `sglang:main` 完全同步（合入 394 个 commits）：

- **全面同步上游 Master**：完整集成上游官方最新的 DSA 协同 Top-K 算子（`csrc/elementwise/topk.hip`）、`KVIndexTranslator` 批量索引映射机制、Dense One-shot Prefill 路径以及推测验证流时序对齐（`plan_stream.wait_stream(fwd_stream)`）。
- **针对 RDNA3 (gfx1100) 硬件特性的深层适配**：
  - **寄存器级反量化**：RDNA3 WMMA 单元原生支持 BF16/FP16。在 Triton Decode 与 Extend 注意力核中通过寄存器级类型转换反量化（`IS_FP8_KV and _IS_RDNA`），避免软仿真 ALU 惩罚。
  - **规避 CDNA 陷阱 / 自定义 All-Reduce**：通过 `-DSGL_IS_RDNA` 宏禁用 CDNA 特有的 MUBUF / 跨卡 Peer-IPC 内存陷阱，回退到经过 RDNA 调优的高效实现与标准 RCCL 通信。
  - **矢量运算符重载支持**：编译 HIP 原生算子时取消定义 `__HIP_NO_HALF_OPERATORS__` 与 `__HIP_NO_HALF_CONVERSIONS__`，确保 Clang++/HIPCC 矢量算术重载无缝通过。
  - **集成 Unified Verify 验证适配器**：保留 RDNA 统一推测验证核（`SGL_RDNA_VLLM_VERIFY=1`），在 4k–16k 深度上下文生成中取得 +8-10% 的持续速度提升。
- **混合 GDN / Mamba 架构推测解码**：深度支持 Qwen 3.8 混合架构（全注意力 + Mamba-2 循环层），在 EAGLE MTP-3 多步推测下通过独立的中间状态暂存区（`intermediate_ssm_state_cache`）防止被拒绝草稿 Token 污染主状态。

---

## 准备模型权重

目标模型使用的是经过微调修复的本地权重 `qwen3.8-27b-mtp-fixed`，派生自 HuggingFace 官方社区开源的 [`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ)：

1. **下载基础模型**：
   ```bash
   huggingface-cli download Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ --local-dir /path/to/qwen3.8-27b-mtp-fixed
   ```
2. **问题根因**：原仓库在 `model_extra_tensors.safetensors` 中将全部 15 个 MTP 张量以纯 **BF16** 格式存储，但在 `config.json` 的 `quantization_config.dynamic` 中声明了正向匹配规则 `"+:.*mtp.*"` 和 `"+:.*mtp\.fc.*"`（4-bit，group 64）。SGLang 和 vLLM 的 GPTQ 权重加载器会信任此配置并尝试按量化格式加载 MTP 层，导致加载报错失败。
3. **修复补丁**：编辑该模型目录下的 `config.json`，在 `quantization_config.dynamic` 中将所有 `"+:"` 规则替换为单条 `"-:.*mtp.*"` 排除规则：
   ```json
   "dynamic": {
     "-:.*mtp.*": {
       "bits": 16,
       "group_size": 128
     }
   }
   ```
   该修改指导加载器跳过 MTP 层的量化解析，直接以原生 BF16 精度加载权重。

---

## 硬件与运行环境

| 组件 | 配置参数 |
|---|---|
| **显卡** | 2x AMD Radeon RX 7900 XTX (每张 24GB GDDR6, Navi 31, gfx1100, TP=2) |
| **总线连接** | PCIe Gen 4.0 x16 (16.0 GT/s，单卡双向带宽约 31.5 GB/s) |
| **CPU / 内存** | AMD Ryzen 5 9600X, DDR5 高速内存 |
| **ROCm 驱动栈** | ROCm 7.14 (HIP 7.14.60850) |
| **PyTorch** | 2.11.0+git (ROCm 7.2 官方构建 wheel) |
| **模型** | `Qwen3.8-27B-W4A16-AutoRound-GPTQ` (`qwen3.8-27b-mtp-fixed`) |
| **注意力后端** | Triton Attention (`triton_backend.py`)，开启 RDNA3 Wave32 切分 |

---

## 构建与安装

### 1. 编译 C++/HIP 原生算子库
```bash
# 激活 ROCm Python 虚拟环境
source .venv-rocm/bin/activate

# 针对 gfx1100 架构编译 sgl_kernel 原生扩展
cd sglang/python/sglang/kernels/aot
PYTORCH_ROCM_ARCH="gfx1100" python setup_rocm.py build_ext --inplace

# 将编译完成的 .so 动态链接库同步至虚拟环境 site-packages
cp python/sgl_kernel/common_ops.*.so $VIRTUAL_ENV/lib/python3.12/site-packages/sgl_kernel/
```

### 2. 安装可编辑 Python 包
```bash
cd sglang/python
pip install --no-build-isolation --no-deps -e .
```

### 3. 编译独立的 RDNA Custom All-Reduce 扩展（可选推荐）
RDNA3 架构缺少 CDNA 专属的 MUBUF / peer-IPC 硬件集合通信，本仓库在 `scripts/rdna_ar` 提供了针对 gfx1100 优化的一阶段点对点 PCIe 极速 All-Reduce 算子：

```bash
# 预编译并将 .so 动态库持久化缓存至 scripts/rdna_ar/vendor/
python sglang/scripts/rdna_ar/rdna_ar_ext.py
```
*注：当设置环境变量 `SGLANG_RDNA_CUSTOM_AR=1` 时，SGLang 会自动查找源码树内的 `sglang/scripts/rdna_ar`（或通过 `SGLANG_RDNA_AR_PATH` 指定自定义路径）。预编译可避免服务冷启动时的 JIT 编译开销。*

---

## 生产启动命令

通过环境变量开启 RDNA3 调优加速，启动支持 EAGLE MTP-3 推测解码与 Radix 前缀缓存的 OpenAI 兼容推理服务：

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

### DFlash2 推测解码（生产配置）

生产环境 systemd 服务（`sglang.service`）通过 `scripts/rdna-serve.sh qwen38-autoround-dflash2`
启动 DFlash2 配方。等效的手动启动命令如下：

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

DFlash2 相比上方 MTP-3 配方的关键差异：

- 草稿模型为独立的 **Qwen3.8-27B-DFlash2** 权重（5 层草稿网络，HF Hub 上的
  `incoai/Qwen3.8-27B-DFlash2`），通过 `--speculative-draft-model-quantization unquant`
  以非量化方式加载，不继承目标模型的 W4A16 量化配置。
- `--speculative-num-draft-tokens 8`（7 token 草稿块 + 1），配合
  `--speculative-draft-window-size 2048` 滑动草稿窗口。
- `--kv-cache-dtype bfloat16` 与 `--mem-fraction-static 0.90` 为草稿模型预留独立
  KV 预算（DFlash2 总显存约 14.5 GiB，MTP-3 约 11.3 GiB）。
- 额外启用 `--chunked-prefill-size 2048` 以及 qwen3 推理 / 工具调用解析器。

本 Fork 实测（2026-09-09，双卡 RX 7900 XTX TP=2）：数学 CoT 生成速度约 166 tok/s
（GSM8K/MATH-500），120k Agentic 会话回放平均生成速度约 83 tok/s，16k 上下文深度
TG 保持率超过 100%。

---

## 实测性能基准对比

全部基准均在双卡 RX 7900 XTX (TP=2) 实机测试，运行 Qwen3.8-27B-W4A16 模型。

### 1. 标准化上下文深度衰减测试 (`llama-benchy`)
*标准 Prompt Prefill ($PP=2048$) 与 Token Generation ($TG=128$), 并发数 = 1*

| 上下文深度 | SGLang MTP-3 (本分支) | vLLM MTP-3 基线 | vLLM DFlash2 基线 | 对比 vLLM MTP-3 优势 |
|:---:|:---:|:---:|:---:|:---:|
| **Depth 0** | **97.1 tok/s** | 88.6 tok/s | 71.1 tok/s | **+9.6%** |
| **Depth 4,096** | **97.3 tok/s** | 83.3 tok/s | 68.4 tok/s | **+16.8%** |
| **Depth 8,192** | **91.5 tok/s** | 93.3 tok/s | 71.8 tok/s | -1.9% |
| **Depth 16,384** | **95.7 tok/s** | 75.9 tok/s | 62.2 tok/s | **+26.1%** |
| **速度留存率 (16k / 0k)** | **98.5%** | 85.7% | 87.5% | **长文本衰减极低** |

### 2. 真实 120k Agent 多轮会话回放（16 轮离散交互）
*采样自真实 120k 长文本 Agent 对话（332 $\to$ 120,443 tokens），启用 Radix 前缀缓存*

| 评估指标 | SGLang MTP-3 (本分支) | vLLM MTP-3 | vLLM DFlash2 | 提升幅度 |
|---|:---:|:---:|:---:|:---:|
| **平均生成速度 (Mean TG)** | **87.89 tok/s** | 63.50 tok/s | 67.96 tok/s | **提速 +38.4%** |
| **中位数速度 (Median TG)** | **85.97 tok/s** | 61.52 tok/s | 67.16 tok/s | **提速 +39.7%** |
| **抖动率 (CV Jitter %)** | **12.92%** | 45.34% | 36.56% | **平稳度提升 3.5 倍** |
| **最差轮次保底速度** | **66.71 tok/s** | 16.94 tok/s | 32.94 tok/s | **最低速度提升 4 倍** |

### 3. 数学思维链推理测试 (GSM8K & MATH-500)
*Greedy 贪婪采样，temperature = 0.0，max_tokens = 1024*

| 数据集用例 | 生成 Token 数量 | 首字延迟 (TTFT) | Prefill 速度 | 生成速度 (TG) | 准确率 |
|---|:---:|:---:|:---:|:---:|:---:|
| **GSM8K #1** | 48 | 0.169s | 533.5 tok/s | **98.9 tok/s** | 100% 正确 |
| **GSM8K #2** | 196 | 0.180s | 626.9 tok/s | **116.2 tok/s** | 100% 正确 |
| **MATH-500 #1** | 169 | 0.149s | 556.4 tok/s | **114.7 tok/s** | 100% 正确 |
| **MATH-500 #2** | 212 | 0.137s | 533.3 tok/s | **111.2 tok/s** | 100% 正确 |
| **综合平均** | — | **0.159s** | **562.5 tok/s** | **110.3 tok/s** | **100% 正确** |

### 4. 多并发吞吐实测 ($c=4$)
*使用 `llama-benchy` 进行多并发请求压测 ($PP=2048, TG=128$, Depth 0)*

| 推理服务引擎 | 并发数 | Prefill 总吞吐 (PP) | 生成总吞吐 (TG) | 峰值生成吞吐 | 运行稳定性与说明 |
|---|:---:|:---:|:---:|:---:|---|
| **SGLang MTP-3 (本分支)** | **c = 4** | **1,974.5 tok/s** | **147.5 tok/s** | **206.0 tok/s** | **100% 稳定运行**，Decode CUDA 图与 MTP-3 正常工作 |
| **vLLM Baseline (无投机)** | c = 4 | 1,891.1 tok/s | 83.1 tok/s | 180.0 tok/s | 原生稳定，但解码速度较低 |
| **vLLM DFlash2** | c = 4 | 1,693.2 tok/s | 75.9 tok/s | 188.0 tok/s | 投机开销导致多并发总 TG 吞吐反而低于 Baseline |
| **vLLM MTP-3** | c = 4 | — | *(崩溃中断)* | — | 多并发 `batch > 1` 时频繁非法显存访问崩溃 |
| **llama.cpp (MTP)** | c = 4 | 636.3 tok/s | 50.1 tok/s | — | 受限于插槽并发队列瓶颈 (`-np 2`) |

> **多并发核心结论**：此前在 ROCm 平台上，vLLM 的投机解码在多并发下普遍面临崩溃或负优化（DFlash2 吞吐不如普通非投机模型，原生 MTP 则直接非法地址异常）。而 SGLang 依靠 Wave32 对齐的 Triton 解码图和独立的 Mamba 状态缓冲，在 4 并发下展现出 **147.5 tok/s 的总生成吞吐**，比 vLLM 基线高出 **+77.5%**，比 vLLM DFlash2 高出 **+94.4%**。

---

## 运维注意事项与避坑指南

> ⚠️ **高危操作提示：切勿在 RDNA3 (gfx1100) 环境执行 `rocm-smi --gpureset`**
> 在 Navi 31 架构上对 GPU 发送驱动级重置指令会导致 PCIe 根端口锁死、全机卡死，只能通过机箱物理冷重启恢复。

1. **FP8 与 BF16 KV Cache 选型**：
   - 尽管 FP8 能节省一半 KV 显存，但在极长上下文的动态追加 Prefill 中容易诱发 Triton 动态索引计算导致的内核故障。
   - 生产级稳定运行请坚持使用 BF16 KV 缓存（`--kv-cache-dtype auto`）。
2. **CUDA 图调度**：
   - 推荐仅对 Decode 阶段录制 CUDA 图（`--cuda-graph-bs-decode 1 2 4`）。在 24GB 显存显卡上切勿启用 Prefill 阶段图捕获，以防启动显存不足。
3. **Mamba 循环状态显存优化**：
   - Mamba 状态缓存按并发数预分配。单人 Agent 使用场景下，设置 `--max-running-requests 4` 与 `--max-mamba-cache-size 20` 可释放单卡约 **2.6 GB 显存**，直接将 BF16 窗口推至 **192k (196,608 tokens)**。

---

## ROCm Linux KFD 事件版本号忙等待修复 (`scripts/rdna_ar/kfd_event_age_fix.c`)

在现代 Linux 内核 (KFD ABI 1.14+) 搭配 ROCm 6.x/7.x 运行时，即使服务完全处于空闲状态无任何流量，系统也常出现**两颗 CPU 核心被 100% 满载打满**的现象。

### 根本原因
1. **SGLang 调度循环轮询**：默认情况下 SGLang 调度主循环持续主动轮询请求。开启 `--sleep-on-idle` 会激活 ZMQ 套接字等待（`IdleSleeper`），将 Python 调度进程在空闲时的 CPU 占用降至 0%。
2. **ROCR 运行时 Event Age 状态脱节 Bug**：开启 `--sleep-on-idle` 后，每张 GPU 仍会有 1 个 ROCm 原生后台线程 (`libhsa-runtime64.so` 中的 `AsyncEventsLoop`) 保持 100% CPU 满载。
   - Linux KFD ABI 1.14+ 引入了单调递增的 `event_age` 机制，用于防止多等待者线程漏掉事件通知。
   - 上游 ROCR 运行时在调用 `AMDKFD_IOC_WAIT_EVENTS` 前，将栈上的 `event_age` 硬编码/重置为 `1`。一旦 GPU 触发过硬件事件，内核内部维护的 `ev->event_age >= 2`。
   - Linux 内核驱动判定 `ev->event_age != last_event_age`，立即将事件置为已就绪 (`KFD_IOC_WAIT_RESULT_COMPLETE`) 并在 3 微秒内返回用户态。
   - 用户态紧接着再次循环，导致单卡每秒产生 **~340 万次系统调用**，单核 CPU 被完全占满。

### 解决方案
我们在 `scripts/rdna_ar/kfd_event_age_fix.c` 提供了零性能开销的动态链接拦截垫片（Interposer Shim）：
- 拦截 `ioctl(AMDKFD_IOC_WAIT_EVENTS)`。
- 在用户态使用无锁缓存记录内核返回的最新 `last_event_age`。
- 在下一次调用内核前自动填入同步后的版本号，使 Linux AMDKFD 内核驱动正确进入深度休眠 (`schedule_timeout`)。
- **效果**：两张 GPU 对应的后台空闲线程 CPU 占用直接由 **100% 降至 0.0%**，无需修改系统软件包或重编 ROCm / PyTorch。

编译垫片：
```bash
make -C scripts/rdna_ar
```

通过环境变量注入：
```bash
export LD_PRELOAD="scripts/rdna_ar/vendor/kfd_event_age_fix.so:$LD_PRELOAD"
```

