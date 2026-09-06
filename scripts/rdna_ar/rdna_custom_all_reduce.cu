// Standalone RDNA3 custom all-reduce wrapper.
//
// Implements the sglang python-facing HIP custom-AR API (init_custom_ar,
// register_buffer, all_reduce_reg, all_reduce_unreg, ...) on top of the
// evolved sglang::CustomAllreduce class in custom_all_reduce_hip.cuh (the
// in-tree custom_all_reduce.hip wrapper is stale relative to that header).
//
// Python-side contract (see srt/distributed/device_communicators/
// custom_all_reduce.py, HIP branch):
//   init_custom_ar(meta, rank_data, handles, offsets, rank, full_nvlink)
//     - meta     : this rank's IPC-shareable signal buffer
//     - handles  : all ranks' meta IPC handles (bytes), offsets all 0
//   register_buffer(fa, t, handles, offsets)
//     - t        : this rank's tensor to share
//     - handles  : all ranks' IPC handles of the tensor base allocation
//   all_reduce_reg(fa, inp, out)            -> one/two-shot reduce (inp must be
//                                              registered via register_buffer)
//   all_reduce_unreg(fa, inp, reg, out)     -> copy inp into registered reg,
//                                              reduce that
//   get_graph_buffer_ipc_meta(fa) -> (Tensor, int64[])
//   register_graph_buffers(fa, handles, offsets)
//   dispose / meta_size / allocate_meta_buffer / get_meta_buffer_ipc_handle

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>
#include <torch/extension.h>

// 12.115 live-context verifier: device-visible debug globals referenced by
// the patched cross_device_reduce_1stage tail (see rdna_ar_ext._patch_header).
// The kernel, when enabled, recomputes out[0..3] from both ranks' registered
// buffers independently and counts mismatches — works through graph replay
// with zero synchronization overhead.
__device__ uint32_t sglang_g_ar_dbg_enabled = 0;
__device__ uint32_t sglang_g_ar_dbg_mismatch = 0;
__device__ uint32_t sglang_g_ar_dbg_calls = 0;

#include "custom_all_reduce_hip.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

static sglang::CustomAllreduce* fa_of(fptr_t _fa) {
  return reinterpret_cast<sglang::CustomAllreduce*>(_fa);
}

void copy_into_reg_buffer(fptr_t _fa, torch::Tensor& inp,
                          torch::Tensor& reg_buffer);

static void open_peer_base(const std::string& handle_bytes, int64_t offset,
                           void** out_ptr) {
  hipIpcMemHandle_t h;
  std::memcpy(&h, handle_bytes.data(), sizeof(h));
  void* base = nullptr;
  CHECK_CUDA_SUCCESS(hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess));
  *out_ptr = (void*)((char*)base + offset);
}

fptr_t init_custom_ar(torch::Tensor& meta, torch::Tensor& rank_data,
                      const std::vector<std::string>& handles,
                      const std::vector<int64_t>& offsets, int64_t rank,
                      bool full_nvlink) {
  int world_size = offsets.size();
  TORCH_CHECK(world_size <= 8, "world size > 8 unsupported");
  TORCH_CHECK(world_size == (int)handles.size(), "handles/offsets mismatch");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  // Enable peer access explicitly before any IPC mapping: the "lazy" peer
  // access implied by hipIpcMemLazyEnablePeerAccess does not reliably wire up
  // compute-kernel access on RDNA3/PCIe on this stack.
  // NOTE: every call here must consume its returned error (HIP records it in
  // the thread's last-error slot; a stale error would poison the next torch
  // op with a bogus AcceleratorError).
  int dev = 0;
  CHECK_CUDA_SUCCESS(hipGetDevice(&dev));
  for (int i = 0; i < world_size; i++) {
    if (i == (int)rank) continue;
    // Initialize the peer device in this process first (TP processes only
    // touch their own GPU; enabling access to an uninitialized device fails
    // with hipErrorInvalidValue).
    // 12.116: SGL_RDNA_NO_PEER_INIT skips the peer-device context creation
    // (hipSetDevice+hipFree(0)) — a second HIP context on the peer GPU
    // inside this process is a candidate poison.
    if (getenv("SGL_RDNA_NO_PEER_INIT") == nullptr) {
      (void)hipSetDevice(i);
      (void)hipFree(0);
      (void)hipSetDevice(dev);
      (void)hipGetLastError();  // clear device-switch leftovers
    } else {
      fprintf(stderr, "[rdna_ar] peer device-init dance SKIPPED (env gate)\n");
    }
    // 12.116 ROOT-CAUSE PROBE: with SGL_RDNA_NO_PEER_ACCESS=1, skip the
    // explicit peer-access enablement. Discriminates "early P2P enablement
    // poisons NCCL's own transports" from the IPC/buffer side of init.
    if (getenv("SGL_RDNA_NO_PEER_ACCESS") != nullptr) {
      fprintf(stderr, "[rdna_ar] peer-access enablement SKIPPED (env gate)\n");
      continue;
    }
    hipError_t e = hipDeviceEnablePeerAccess(i, 0);
    (void)hipGetLastError();  // always consume; failure is non-fatal
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      // Peer access may be unavailable (e.g. IOMMU); IPC open will still try.
      fprintf(stderr, "[rdna_ar] hipDeviceEnablePeerAccess(%d) from %d: %s "
              "(continuing with lazy IPC mapping)\n", i, dev,
              hipGetErrorString(e));
    }
  }

  sglang::Signal* signals[8];
  for (int i = 0; i < world_size; i++) {
    if (i == (int)rank) {
      signals[i] = reinterpret_cast<sglang::Signal*>(meta.data_ptr());
    } else if (getenv("SGL_RDNA_SKIP_META_IPC") != nullptr) {
      // 12.116: never dereferenced while FORCE_NCCL keeps AR kernels idle.
      fprintf(stderr, "[rdna_ar] peer meta IPC open SKIPPED (env gate)\n");
      signals[i] = nullptr;
    } else {
      void* p = nullptr;
      open_peer_base(handles[i], offsets[i], &p);
      signals[i] = reinterpret_cast<sglang::Signal*>(p);
    }
  }
  return (fptr_t) new sglang::CustomAllreduce(
      signals, rank_data.data_ptr(), rank_data.numel(), (int)rank, world_size,
      full_nvlink);
}

bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

void _all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                 hipStream_t stream) {
  auto fa = fa_of(_fa);
  TORCH_CHECK(_is_weak_contiguous(out));
  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(inp.data_ptr()),
                           reinterpret_cast<float*>(out.data_ptr()), out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(inp.data_ptr()),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    }
    case at::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
    default:
      throw std::runtime_error(
          "custom allreduce supports float32, float16, bfloat16 only");
  }
}

void all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  _all_reduce(_fa, inp, out, stream);
}

void all_reduce_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer,
                      torch::Tensor& out) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  auto input_size = inp.numel() * inp.element_size();
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(input_size <= reg_buffer.numel() * reg_buffer.element_size(),
              "registered buffer too small for input");
  copy_into_reg_buffer(_fa, inp, reg_buffer);
  _all_reduce(_fa, reg_buffer, out, stream);
}

// Fused all-reduce + residual-add + RMSNorm (RDNA standalone impl, ws=2).
// Copies inp into the registered buffer (system-scope, capture-safe), then
// launches cross_device_reduce_1stage_rmsnorm. residual is read locally
// (TP-replicated); weight is the RMSNorm gamma [hidden].
void fused_allreduce_rmsnorm(fptr_t _fa, torch::Tensor& inp,
                             torch::Tensor& residual, torch::Tensor& weight,
                             double eps, torch::Tensor& reg_buffer,
                             torch::Tensor& out_normed,
                             torch::Tensor& out_residual) {
  // JOURNAL 12.114: the v13 fused kernel (cross_device_reduce_1stage_rmsnorm)
  // was authored as an UNCOMMITTED edit to the AOT header and lost in the
  // upstream rebase; the /tmp build cache was wiped by the 2026-09-01
  // reboots, so this build cannot link the fused member. Throw so the
  // python side (parallel_state.fused_allreduce_rmsnorm) falls back to
  // custom-AR + separate RMSNorm — recovering the custom-AR win (12.5)
  // while the fused kernel is re-authored (12.13 design notes + surviving
  // test harness scripts/rdna_ar/test_fused_ar_rms.py).
  throw std::runtime_error(
      "fused AR+RMSNorm: v13 kernel lost (header wipe); custom AR still "
      "active — python falls back to AR + separate RMSNorm");
}
#if 0
void fused_allreduce_rmsnorm_impl_original(fptr_t _fa, torch::Tensor& inp,
                             torch::Tensor& residual, torch::Tensor& weight,
                             double eps, torch::Tensor& reg_buffer,
                             torch::Tensor& out_normed,
                             torch::Tensor& out_residual) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  TORCH_CHECK(inp.scalar_type() == out_normed.scalar_type() &&
                  inp.scalar_type() == out_residual.scalar_type() &&
                  inp.scalar_type() == residual.scalar_type() &&
                  inp.scalar_type() == weight.scalar_type(),
              "fused AR+RMSNorm dtype mismatch");
  TORCH_CHECK(inp.dim() == 2 && residual.is_contiguous() &&
                  out_normed.is_contiguous() && out_residual.is_contiguous(),
              "fused AR+RMSNorm expects 2D contiguous tensors");
  const int64_t tokens = inp.size(0);
  const int64_t hidden = inp.size(1);
  TORCH_CHECK(residual.numel() == inp.numel() &&
                  out_normed.numel() == inp.numel() &&
                  out_residual.numel() == inp.numel() &&
                  weight.numel() == hidden,
              "fused AR+RMSNorm shape mismatch");
  const int64_t input_size = inp.numel() * inp.element_size();
  TORCH_CHECK(input_size <= reg_buffer.numel() * reg_buffer.element_size(),
              "registered buffer too small for fused AR+RMSNorm input");
  copy_into_reg_buffer(_fa, inp, reg_buffer);
  auto input_ptr = (char*)reg_buffer.data_ptr();
  switch (inp.scalar_type()) {
    case at::ScalarType::Float:
      throw std::runtime_error("fused AR+RMSNorm: fp32 unsupported");
    case at::ScalarType::Half:
      fa_of(_fa)->allreduce_rmsnorm<half>(
          stream, (half*)input_ptr, (half*)residual.data_ptr(),
          (half*)weight.data_ptr(), (half*)out_normed.data_ptr(),
          (half*)out_residual.data_ptr(), (float)eps, (int)tokens, (int)hidden);
      break;
    case at::ScalarType::BFloat16:
      fa_of(_fa)->allreduce_rmsnorm<nv_bfloat16>(
          stream, (nv_bfloat16*)input_ptr, (nv_bfloat16*)residual.data_ptr(),
          (nv_bfloat16*)weight.data_ptr(), (nv_bfloat16*)out_normed.data_ptr(),
          (nv_bfloat16*)out_residual.data_ptr(), (float)eps, (int)tokens,
          (int)hidden);
      break;
    default:
      throw std::runtime_error("fused AR+RMSNorm: unsupported dtype");
  }
}
#endif  // JOURNAL 12.114: disabled original fused body (v13 kernel lost)

void dispose(fptr_t _fa) { delete fa_of(_fa); }

// Debug: return the device pointers this rank's RankData holds for all ranks
// (ptrs[i] = pointer used when reading rank i's data), plus self/peer signal
// base pointers. Vector layout: [ptr0, ptr1, ..., self_sg, sg0, sg1].
std::vector<int64_t> debug_ptrs(fptr_t _fa) {
  auto fa = fa_of(_fa);
  sglang::RankData rd;
  CHECK_CUDA_SUCCESS(hipMemcpy(&rd, fa->d_rank_data_base_ - 1, sizeof(rd),
                                hipMemcpyDeviceToHost));
  std::vector<int64_t> out;
  for (int i = 0; i < fa->world_size_; i++) out.push_back((int64_t)rd.ptrs[i]);
  out.push_back((int64_t)fa->self_sg_);
  for (int i = 0; i < fa->world_size_; i++)
    out.push_back((int64_t)fa->sg_.signals[i]);
  return out;
}

// Debug: kernel-side read of the first n bf16 elements of rd.ptrs[idx].
__global__ void debug_read_kernel(void* const* ptrs, int idx, void* out, int n) {
  const __hip_bfloat16* src =
      reinterpret_cast<const __hip_bfloat16*>(ptrs[idx]);
  __hip_bfloat16* dst = reinterpret_cast<__hip_bfloat16*>(out);
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

torch::Tensor debug_kernel_read(fptr_t _fa, int64_t idx, int64_t n) {
  auto fa = fa_of(_fa);
  sglang::RankData* d_rd = fa->d_rank_data_base_ - 1;
  auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  torch::Tensor out = torch::zeros({n}, options);
  debug_read_kernel<<<1, 64>>>((void* const*)d_rd, (int)idx, out.data_ptr(),
                               (int)n);
  CHECK_CUDA_SUCCESS(hipStreamSynchronize(
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()));
  return out;
}

int64_t meta_size() { return sizeof(sglang::Signal); }

// 12.115 debug verifier controls: enable the in-kernel check and read/reset
// the mismatch+call counters (device globals, no stream dependency).
// ROCm 7.x: HIP_SYMBOL(X) is identity; host code takes &device_var directly.
static void set_dbg(const void* sym, uint32_t v) {
  CHECK_CUDA_SUCCESS(hipMemcpyToSymbol(sym, &v, sizeof(v)));
}
void ar_dbg_set(int64_t enabled) {
  set_dbg(&sglang_g_ar_dbg_enabled, (uint32_t)enabled);
}
std::tuple<int64_t, int64_t> ar_dbg_pop() {
  uint32_t mm = 0, calls = 0;
  CHECK_CUDA_SUCCESS(hipMemcpyFromSymbol(
      &mm, (const void*)&sglang_g_ar_dbg_mismatch, sizeof(mm)));
  CHECK_CUDA_SUCCESS(hipMemcpyFromSymbol(
      &calls, (const void*)&sglang_g_ar_dbg_calls, sizeof(calls)));
  set_dbg(&sglang_g_ar_dbg_mismatch, 0);
  set_dbg(&sglang_g_ar_dbg_calls, 0);
  return {(int64_t)mm, (int64_t)calls};
}

// Copy-in with cache-bypassing stores. PCIe peer reads on this stack only
// observe HBM, never the writer's L2 dirty lines; small hipMemcpyAsync D2D
// copies run as a compute kernel with normal (L2 write-back) stores, so the
// peer reads zeros. RCCL solves the same problem with sc0/sc1 ("system
// coherent") bits on its data accesses — the same trick as the AR header's
// __builtin_amdgcn_global_{load,store}_b128 with an empty (system) syncscope.
typedef unsigned int sgl_v4u __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) sgl_v4u* sgl_v4u_gptr;

__global__ void copy_sys_kernel(const sgl_v4u* __restrict__ src,
                                sgl_v4u* __restrict__ dst, long long n16) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < n16; i += stride) {
    sgl_v4u v = src[i];  // local read, normal path is fine
    __builtin_amdgcn_global_store_b128((sgl_v4u_gptr)(dst + i), v, "");
  }
}

void copy_into_reg_buffer(fptr_t _fa, torch::Tensor& inp,
                          torch::Tensor& reg_buffer) {
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  auto nbytes = inp.numel() * inp.element_size();
  TORCH_CHECK(nbytes % 16 == 0, "copy size must be 16B multiple");
  long long n16 = nbytes / 16;
  int threads = 256;
  int blocks = (int)std::min<long long>(64, (n16 + threads - 1) / threads);
  hipLaunchKernelGGL(copy_sys_kernel, dim3(blocks), dim3(threads), 0, stream,
                     (const sgl_v4u*)inp.data_ptr(),
                     (sgl_v4u*)reg_buffer.data_ptr(), n16);
}

void register_buffer(fptr_t _fa, torch::Tensor& t,
                     const std::vector<std::string>& handles,
                     const std::vector<int64_t>& offsets) {
  auto fa = fa_of(_fa);
  int world_size = fa->world_size_;
  TORCH_CHECK(world_size == (int)handles.size(), "handles count mismatch");
  void* ptrs[8];
  for (int i = 0; i < world_size; i++) {
    if (i == fa->rank_) {
      ptrs[i] = t.data_ptr();
    } else {
      void* p = nullptr;
      open_peer_base(handles[i], offsets[i], &p);
      ptrs[i] = p;
    }
  }
  fa->register_buffer(ptrs);
}

std::tuple<torch::Tensor, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    fptr_t _fa) {
  auto [handle_bytes, offsets] = fa_of(_fa)->get_graph_buffer_ipc_meta();
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  torch::Tensor handles = torch::empty(
      {(int64_t)handle_bytes.size()}, options);
  std::memcpy(handles.data_ptr(), handle_bytes.data(), handle_bytes.size());
  return {handles, std::move(offsets)};
}

void register_graph_buffers(fptr_t _fa, const std::vector<std::string>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  fa_of(_fa)->register_graph_buffers(handles, offsets);
}

void free_meta_buffer(void* buffer) { CHECK_CUDA_SUCCESS(hipFree(buffer)); }

torch::Tensor get_meta_buffer_ipc_handle(torch::Tensor& inp) {
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  torch::Tensor data_handle = torch::empty(
      {(int64_t)sizeof(hipIpcMemHandle_t)}, options);
  hipError_t e = hipIpcGetMemHandle((hipIpcMemHandle_t*)data_handle.data_ptr(),
                                    inp.data_ptr());
  if (e != hipSuccess) {
    fprintf(stderr, "[rdna_ar] hipIpcGetMemHandle failed on ptr=%p: %s "
            "(allocated with hipExtMallocWithFlags uncached)\n", inp.data_ptr(),
            hipGetErrorString(e));
    (void)hipGetLastError();
    throw std::runtime_error(
        std::string("hipIpcGetMemHandle failed: ") + hipGetErrorString(e));
  }
  return data_handle;
}

torch::Tensor allocate_meta_buffer(int64_t size) {
  auto device_index = c10::hip::current_device();
  fprintf(stderr, "[rdna_ar] allocate_meta_buffer size=%ld on device %d\n",
          (long)size, device_index);
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  CHECK_CUDA_SUCCESS(hipThreadExchangeStreamCaptureMode(&mode));
  hipError_t alloc_e = hipExtMallocWithFlags((void**)&buffer, size,
                                             hipDeviceMallocUncached);
  if (alloc_e != hipSuccess) {
    fprintf(stderr, "[rdna_ar] hipExtMallocWithFlags failed: %s\n",
            hipGetErrorString(alloc_e));
    (void)hipGetLastError();
    throw std::runtime_error(
        std::string("hipExtMallocWithFlags failed: ") + hipGetErrorString(alloc_e));
  }
  fprintf(stderr, "[rdna_ar] meta buffer allocated at %p\n", buffer);
  CHECK_CUDA_SUCCESS(hipMemsetAsync(buffer, 0, size, stream));
  CHECK_CUDA_SUCCESS(hipStreamSynchronize(stream));
  CHECK_CUDA_SUCCESS(hipThreadExchangeStreamCaptureMode(&mode));
  auto options = torch::TensorOptions()
                     .dtype(torch::kI8)
                     .device(torch::kCUDA, device_index);
  return torch::from_blob(buffer, {size}, free_meta_buffer, options);
}

// Dedicated uncached allocation (same path as the signal buffers, which are
// proven visible cross-GPU) — torch caching-allocator segments behave
// differently for PCIe peer reads on this stack.
void free_reg_buffer(void* buffer) { CHECK_CUDA_SUCCESS(hipFree(buffer)); }

torch::Tensor allocate_reg_buffer(int64_t size) {
  auto device_index = c10::hip::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  hipError_t alloc_e = hipExtMallocWithFlags((void**)&buffer, size,
                                             hipDeviceMallocUncached);
  fprintf(stderr, "[rdna_ar] allocate_reg_buffer size=%ld on dev %d -> %p "
          "(rc=%s)\n", (long)size, device_index, buffer,
          hipGetErrorString(alloc_e));
  (void)hipGetLastError();
  if (alloc_e != hipSuccess)
    throw std::runtime_error(
        std::string("hipExtMallocWithFlags(reg) failed: ") + hipGetErrorString(alloc_e));
  // self-test: can we take an IPC handle on this allocation right away?
  hipIpcMemHandle_t probe_handle;
  hipError_t he = hipIpcGetMemHandle(&probe_handle, buffer);
  fprintf(stderr, "[rdna_ar] immediate hipIpcGetMemHandle(reg) rc=%s\n",
          hipGetErrorString(he));
  (void)hipGetLastError();
  CHECK_CUDA_SUCCESS(hipMemset(buffer, 0, size));
  auto options = torch::TensorOptions()
                     .dtype(torch::kI8)
                     .device(torch::kCUDA, device_index);
  return torch::from_blob(buffer, {size}, free_reg_buffer, options);
}

// 12.116 canary helper: allocate the reg buffer PLUS a guard tail in ONE
// uncached hip allocation, fill the guard with 0xAB. Any kernel that writes
// past the reg buffer's end lands in the guard (adjacent by construction).
torch::Tensor allocate_reg_buffer_guarded(int64_t size, int64_t guard) {
  auto device_index = c10::hip::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  CHECK_CUDA_SUCCESS(hipExtMallocWithFlags((void**)&buffer, size + guard,
                                           hipDeviceMallocUncached));
  CHECK_CUDA_SUCCESS(hipMemset(buffer, 0, size));
  CHECK_CUDA_SUCCESS(hipMemset((char*)buffer + size, 0xAB, guard));
  auto options = torch::TensorOptions()
                     .dtype(torch::kI8)
                     .device(torch::kCUDA, device_index);
  return torch::from_blob(buffer, {size + guard}, free_reg_buffer, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar, "init custom allreduce");
  m.def("all_reduce_reg", &all_reduce_reg, "allreduce (registered input)");
  m.def("all_reduce_unreg", &all_reduce_unreg, "allreduce (copy into registered buffer)");
  m.def("fused_allreduce_rmsnorm", &fused_allreduce_rmsnorm, "fused allreduce + residual + rmsnorm (ws=2)");
  m.def("register_buffer", &register_buffer, "register shared buffer");
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "graph buffer ipc meta");
  m.def("register_graph_buffers", &register_graph_buffers, "register graph buffers");
  m.def("allocate_meta_buffer", &allocate_meta_buffer, "allocate meta (signal) buffer");
  m.def("get_meta_buffer_ipc_handle", &get_meta_buffer_ipc_handle, "meta buffer ipc handle");
  m.def("meta_size", &meta_size, "meta size");
  m.def("ar_dbg_set", &ar_dbg_set, "enable/disable in-kernel AR verify");
  m.def("ar_dbg_pop", &ar_dbg_pop, "read+reset (mismatches, calls) counters");
  m.def("dispose", &dispose, "dispose");
  m.def("debug_ptrs", &debug_ptrs, "debug: registered ptrs + signal bases");
  m.def("debug_kernel_read", &debug_kernel_read, "debug: kernel-side read of registered ptr");
  m.def("copy_into_reg_buffer", &copy_into_reg_buffer, "system-scope copy into reg buffer");
  m.def("allocate_reg_buffer", &allocate_reg_buffer, "dedicated uncached reg buffer");
  m.def("allocate_reg_buffer_guarded", &allocate_reg_buffer_guarded,
        "reg buffer + 0xAB guard tail (canary for overrun detection)");
}
