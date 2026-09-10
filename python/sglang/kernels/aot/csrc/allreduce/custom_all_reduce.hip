// HIP wrapper over the evolved sglang::CustomAllreduce class in
// custom_all_reduce_hip.cuh. This is the AOT-built counterpart of
// scripts/rdna_ar/rdna_custom_all_reduce.cu (the standalone JIT extension
// used on RDNA): it exposes the same python-facing surface the
// common_extension_rocm.cc registrations and sgl_kernel.allreduce (HIP
// flavor) expect, with the old 7-arg ctor / handles-opening register_buffer
// API retired (round-2 review H9/N6). The RDNA-only pieces of the standalone
// wrapper (debug verifier globals, fused AR+RMSNorm stub, dedicated uncached
// reg-buffer allocation) are intentionally absent: they are not part of the
// torch.ops surface below, and the PCIe L2-visibility workaround in
// all_reduce_unreg targets MI-class interconnects, where plain D2D copies are
// coherent.
//
// Python-side contract (see srt/distributed/device_communicators/
// custom_all_reduce.py, HIP branch):
//   init_custom_ar(meta, rank_data, handles, offsets, rank, full_nvlink)
//     - meta     : this rank's IPC-shareable signal buffer
//     - handles  : all ranks' meta IPC handles (bytes), offsets all 0
//   register_buffer(fa, t, handles, offsets)
//     - t        : this rank's tensor to share
//     - handles  : all ranks' IPC handles of the tensor base allocation

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "custom_all_reduce_hip.cuh"

// fake pointer type, must match fptr_t type in ops.h
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

static sglang::CustomAllreduce* fa_of(fptr_t _fa) {
  return reinterpret_cast<sglang::CustomAllreduce*>(_fa);
}

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
  TORCH_CHECK(world_size % 2 == 0, "odd number of gpus is not supported");
  TORCH_CHECK(world_size == (int)handles.size(), "handles/offsets mismatch");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  // Enable peer access explicitly before any IPC mapping: the "lazy" peer
  // access implied by hipIpcMemLazyEnablePeerAccess does not reliably wire up
  // compute-kernel access on every stack (observed on RDNA3/PCIe; harmless on
  // xGMI). Every call here must consume its returned error so a stale
  // thread-local error cannot poison the next torch op.
  int dev = 0;
  CHECK_CUDA_SUCCESS(hipGetDevice(&dev));
  for (int i = 0; i < world_size; i++) {
    if (i == (int)rank) continue;
    // Initialize the peer device in this process first (TP processes only
    // touch their own GPU; enabling access to an uninitialized device fails
    // with hipErrorInvalidValue).
    (void)hipSetDevice(i);
    (void)hipFree(0);
    (void)hipSetDevice(dev);
    (void)hipGetLastError();  // clear device-switch leftovers
    hipError_t e = hipDeviceEnablePeerAccess(i, 0);
    (void)hipGetLastError();  // always consume; failure is non-fatal
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      // Peer access may be unavailable (e.g. IOMMU); IPC open will still try.
      fprintf(stderr,
              "custom allreduce: hipDeviceEnablePeerAccess(%d) from %d: %s "
              "(continuing with lazy IPC mapping)\n",
              i, dev, hipGetErrorString(e));
    }
  }

  sglang::Signal* signals[8];
  for (int i = 0; i < world_size; i++) {
    if (i == (int)rank) {
      signals[i] = reinterpret_cast<sglang::Signal*>(meta.data_ptr());
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

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 */
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
                           reinterpret_cast<float*>(out.data_ptr()),
                           out.numel());
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
          "custom allreduce only supports float32, float16 and bfloat16");
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
              "registered buffer is too small to contain the input");
  AT_CUDA_CHECK(hipMemcpyAsync(reg_buffer.data_ptr(), inp.data_ptr(),
                               input_size, hipMemcpyDeviceToDevice, stream));
  _all_reduce(_fa, reg_buffer, out, stream);
}

void dispose(fptr_t _fa) { delete fa_of(_fa); }

int64_t meta_size() { return sizeof(sglang::Signal); }

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
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto handles =
      torch::empty({static_cast<int64_t>(handle_bytes.size())}, options);
  std::memcpy(handles.data_ptr(), handle_bytes.data(), handle_bytes.size());
  return {handles, std::move(offsets)};
}

void register_graph_buffers(fptr_t _fa, const std::vector<std::string>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  fa_of(_fa)->register_graph_buffers(handles, offsets);
}

void free_meta_buffer(void* buffer) { CHECK_CUDA_SUCCESS(hipFree(buffer)); }

torch::Tensor get_meta_buffer_ipc_handle(torch::Tensor& inp) {
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto data_handle =
      torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))}, options);
  hipError_t e = hipIpcGetMemHandle((hipIpcMemHandle_t*)data_handle.data_ptr(),
                                    inp.data_ptr());
  if (e != hipSuccess) {
    // Consume the thread-local error before throwing; a stale recorded error
    // would surface as a bogus AcceleratorError on the next torch op.
    (void)hipGetLastError();
    throw std::runtime_error(
        std::string("hipIpcGetMemHandle failed: ") + hipGetErrorString(e));
  }
  return data_handle;
}

torch::Tensor allocate_meta_buffer(int64_t size) {
  auto device_index = c10::hip::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  CHECK_CUDA_SUCCESS(hipThreadExchangeStreamCaptureMode(&mode));
  hipError_t alloc_e =
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached);
  if (alloc_e != hipSuccess) {
    (void)hipGetLastError();
    throw std::runtime_error(
        std::string("hipExtMallocWithFlags failed: ") +
        hipGetErrorString(alloc_e));
  }
  CHECK_CUDA_SUCCESS(hipMemsetAsync(buffer, 0, size, stream));
  CHECK_CUDA_SUCCESS(hipStreamSynchronize(stream));
  CHECK_CUDA_SUCCESS(hipThreadExchangeStreamCaptureMode(&mode));
  auto options = torch::TensorOptions()
                     .dtype(torch::kI8)
                     .device(torch::kCUDA, device_index);
  return torch::from_blob(buffer, {size}, free_meta_buffer, options);
}
