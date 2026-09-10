# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import ActivationOrdering

# yapf conflicts with isort for this block
# yapf: disable
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
    permute_param_layout_,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    MarlinLinearLayerConfig,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx,
    marlin_zero_points,
)
from sglang.srt.layers.quantization.utils import (
    get_scalar_types,
    replace_parameter,
    unpack_cols,
)
from sglang.srt.utils import is_cuda, is_hip

# The marlin repack op is available on both CUDA and HIP builds; do NOT add
# unrelated CUDA-only gating to this flag.
_has_marlin_repack = is_cuda() or is_hip()

if _has_marlin_repack:
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack

# RDNA3 (gfx1100) has no marlin kernel build; the compressed-tensors W4A16
# path repacks into the exllama/GPTQ layout served by the sgl_kernel GPTQ
# GEMM instead — the kernel family the GPTQ quantization scheme already
# runs on this hardware.
_is_rdna3 = False
if is_hip():
    try:
        from sglang.srt.utils.common import is_rdna_supported

        _is_rdna3 = is_rdna_supported()
    except Exception:
        _is_rdna3 = False

if _is_rdna3:
    from sgl_kernel import gptq_gemm as _rdna3_gptq_gemm
    from sgl_kernel import gptq_shuffle as _rdna3_gptq_shuffle


ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128
}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsLinearScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None,
                 symmetric: Optional[bool] = True,
                 actorder: Optional[ActivationOrdering] = None):

        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError("Marlin kernels require group quantization or "
                             "channelwise quantization, but found no group "
                             "size and strategy is not channelwise.")

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

        self.quant_type = (WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
                           if not self.symmetric else
                           WNA16_SUPPORTED_TYPES_MAP[num_bits])

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, output_size: int,
                       input_size: int, output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        output_size_per_partition = sum(output_partition_sizes)

        if _is_rdna3:
            # RDNA3 GPTQ-layout repack guards (REVIEW 2026-09-10 H13): the
            # repack indexes scales per group along K and derives g_idx via a
            # plain argsort, which two legal compressed-tensors configs cannot
            # express. Channelwise would drive groups = K // -1 negative in
            # _process_weights_after_loading_rdna3; act-order + row-parallel
            # shards scales along K while g_idx keeps full-K group ids ->
            # silent out-of-range scale indexing. Mirror vLLM's kernel rejects
            # loudly instead.
            if self.group_size == -1:
                raise ValueError(
                    "Channelwise (group_size=-1) compressed-tensors weights "
                    "are not supported by the RDNA3 W4A16 kernel. "
                    "Re-quantize with a grouped scheme (e.g. group_size=128)."
                )
            if self.has_g_idx and input_size != input_size_per_partition:
                raise ValueError(
                    "compressed-tensors act-order weights are not supported "
                    "with tensor parallelism > 1 on RDNA3 (scale partitioning "
                    "is incompatible with full-K g_idx). Use TP=1, disable "
                    "act-order, or use the marlin path."
                )

        self.kernel_config = MarlinLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx
        )

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        # RDNA3 consumes the GPTQ kernel layout, which indexes scales and
        # zero points per rank; partition them like the GPTQ scheme instead
        # of repeating full-K scales on every rank (the marlin convention).
        partition_scales = (
            not marlin_repeat_scales_on_all_ranks(
                self.has_g_idx, self.group_size, row_parallel)
            or _is_rdna3
        )

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition //
                                         self.pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }

        zeros_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            )
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(output_dim=0,
                                               packed_dim=0,
                                               packed_factor=self.pack_factor,
                                               **zeros_args)
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
            if not self.symmetric:
                qzeros = PackedvLLMParameter(input_dim=1,
                                             output_dim=0,
                                             packed_dim=0,
                                             packed_factor=self.pack_factor,
                                             **zeros_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
                                            input_dim=0,
                                            weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point"
        self.w_gidx_name = "weight_g_idx"

        device = getattr(layer, self.w_q_name).device
        c = self.kernel_config

        if _is_rdna3:
            self._process_weights_after_loading_rdna3(layer)
            return

        check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace(device)

        def _transform_param(
            layer: torch.nn.Module, name: Optional[str], fn: Callable
        ) -> None:
            if name is not None and getattr(layer, name, None) is not None:

                old_param = getattr(layer, name)
                new_param = fn(old_param)
                # replace the parameter with torch.nn.Parameter for TorchDynamo
                # compatibility
                replace_parameter(
                    layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
                )

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                num_bits=c.weight_type.size_bits,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                group_size=c.group_size,
            )
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            _transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            _transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        _transform_param(layer, self.w_q_name, transform_w_q)
        _transform_param(layer, self.w_s_name, transform_w_s)

    def _process_weights_after_loading_rdna3(self, layer: torch.nn.Module) -> None:
        """gfx1100: repack compressed-tensors tensors for sgl_kernel.gptq_gemm.

        Mirrors vLLM's RDNA3W4A16LinearKernel: permute the pack-quantized
        tensors to the [K/8, N] exllama layout, synthesize zero points for
        symmetric checkpoints, and nibble-shuffle the packed weights.
        """
        c = self.kernel_config
        device = getattr(layer, self.w_q_name).device

        if c.has_g_idx:
            g_idx = torch.argsort(getattr(layer, self.w_gidx_name).data).to(torch.int)
            replace_parameter(layer, self.w_gidx_name, g_idx)
        else:
            g_idx = torch.empty((0,), dtype=torch.int, device=device)
            setattr(layer, self.w_gidx_name,
                    torch.nn.Parameter(g_idx, requires_grad=False))

        if c.zero_points:
            zp = getattr(layer, self.w_zp_name)
            permute_param_layout_(zp, input_dim=0, output_dim=1, packed_dim=1)
            replace_parameter(layer, self.w_zp_name, zp.data.contiguous())
        else:
            # Symmetric checkpoints carry no zero points. The kernel reads
            # zero+1 (GPTQv1 quirk), so the neutral fill for uint4b8's +8
            # bias is 7: dequant(code) = scale * (code - 8).
            fill = 0
            for i in range(self.pack_factor):
                fill |= (c.weight_type.bias - 1) << (4 * i)
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]
            zeros = torch.full(
                (groups, out_features // self.pack_factor),
                fill,
                dtype=torch.int32,
                device=device,
            )
            setattr(layer, self.w_zp_name,
                    torch.nn.Parameter(zeros, requires_grad=False))

        # [out, K/8] packed along K -> [K/8, out], then the exllama nibble
        # shuffle (empty g_idx => identity permutation).
        w_q = getattr(layer, self.w_q_name)
        permute_param_layout_(w_q, input_dim=0, output_dim=1, packed_dim=0)
        w_q = w_q.data.contiguous()
        _rdna3_gptq_shuffle(w_q, g_idx, c.weight_type.size_bits)
        replace_parameter(layer, self.w_q_name, w_q)

        # [out, groups] -> [groups, out]
        w_s = getattr(layer, self.w_s_name)
        permute_param_layout_(w_s, input_dim=0, output_dim=1)
        replace_parameter(layer, self.w_s_name, w_s.data.contiguous())

        # E1 (review follow-up): CT W4A16 checkpoints store scales natively
        # as F16 but they are materialized bf16 under bf16 serving; keep
        # fp16 scales so apply_weights takes the v_dot2 fp16 GPTQ path,
        # mirroring GPTQLinearKernel.process_weights_after_loading. Per
        # layer by construction (this only touches this layer's param).
        # SGLANG_RDNA_FP16_GPTQ=0 disables.
        new_w_s = getattr(layer, self.w_s_name)
        if (
            os.environ.get("SGLANG_RDNA_FP16_GPTQ", "1") == "1"
            and new_w_s.dtype == torch.bfloat16
        ):
            replace_parameter(
                layer, self.w_s_name, new_w_s.data.to(torch.float16).contiguous()
            )
            if not getattr(self.__class__, "_fp16_scales_logged", False):
                self.__class__._fp16_scales_logged = True
                logger.info(
                    "RDNA3 fp16 compressed-tensors W4A16 compute enabled "
                    "(E1): fp16 scales, bf16 activations cast per call, "
                    "output cast back."
                )

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.kernel_config

        def _get_weight_params(
            layer: torch.nn.Module,
        ) -> tuple[
            torch.Tensor,  # w_q
            torch.Tensor,  # w_s
            Optional[torch.Tensor],  # w_zp,
            Optional[torch.Tensor],  # w_gidx
        ]:
            return (
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                getattr(layer, self.w_zp_name or "", None),
                getattr(layer, self.w_gidx_name or "", None),
            )

        w_q, w_s, w_zp, w_gidx = _get_weight_params(layer)

        if _is_rdna3:
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)
            # fp16 path iff this layer's scales are fp16 (E1, set per layer
            # in _process_weights_after_loading_rdna3); no-op when serving
            # dtype is fp16.
            gemm_x = x_2d
            if w_s.dtype == torch.float16 and x_2d.dtype != torch.float16:
                gemm_x = x_2d.to(torch.float16)
            output = _rdna3_gptq_gemm(
                gemm_x,
                w_q,
                w_zp,
                w_s,
                w_gidx,
                True,  # use_shuffle: weights are pre-shuffled (exllama layout)
                c.weight_type.size_bits,
            )
            if gemm_x is not x_2d:
                output = output.to(x.dtype)
            if bias is not None:
                output = output + bias
            return output.reshape(out_shape)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            bias=bias,
        )
