// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 dequant primitives for RDNA3 (gfx1100/gfx1101/gfx1102), templated on
// the activation/scale dtype (half or __hip_bfloat16). The fp16 path reuses
// the classic exllamav2 bit-trick:
//
//   (qa & 0x000F000F) | 0x64006400  ->  half2(1024+q_lo, 1024+q_hi)
//   (qa & 0x00F000F0) | 0x64006400  ->  half2(1024+q_lo*16, 1024+q_hi*16)
//
// The "*16 then divide by 16 in the FMA" trick for the upper-nibble pairs
// works in fp16 because the mantissa (10 bits) is wide enough to hold a value
// shifted by 4 bits. In bf16 the mantissa is only 7 bits, so shifting an upper
// nibble into bits [7:4] would spill into the exponent. To avoid that, the
// bf16 path shifts each pair of nibbles down to bits [3:0]/[19:16] with a
// single right-shift before the OR with 0x43004300 (= bf162(128, 128)).

#ifndef _qdq_4_rdna3_cuh
#define _qdq_4_rdna3_cuh

#include <cstdint>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace gptq_rdna3 {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;

// Bit-shuffle for an int32 holding 8 sequential 4-bit weights q[0..7]:
//   in:  q[7] q[6] q[5] q[4] q[3] q[2] q[1] q[0]   (LSB first)
//   out: q[7] q[5] q[3] q[1] q[6] q[4] q[2] q[0]   (even/odd interleaved)
//
// After shuffle, q[2k]   sits at bits [4k   : 4k+3]   (lower 16)
//                q[2k+1] sits at bits [16+4k: 16+4k+3] (upper 16)
// so a single mask 0x000F000F selects the matching even/odd pair, ready to
// bitcast to half2 / bfloat162 after OR-ing with the magic constant.
__forceinline__ __device__ void shuffle_4bit_8(uint32_t* q) {
  uint32_t qa = q[0];
  uint32_t qb = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0F;
    uint32_t qa1 = (qa & 0xF0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

// ---------------------------------------------------------------------------
// fp16 path
// ---------------------------------------------------------------------------

// Numerics: exact integer subtraction first ((1024+q) - (1024+zero) = q - zero)
// followed by multiplication with scale. This avoids rounding intermediate
// offsets to fp16 before FMA and produces bit-identical weights to the WMMA
// kernel.
__forceinline__ __device__ void prep_zero_scale_fp16(uint32_t zero, half scale,
                                                     half2& z_prep,
                                                     half2& y_prep) {
  union {
    uint16_t u;
    half h;
  } zu;
  zu.u = (uint16_t)(0x6400 | zero);
  z_prep = __half2half2(zu.h);
  y_prep = __half2half2(scale);
}

__forceinline__ __device__ void prep_zero_scale_fp16_precise(uint32_t zero,
                                                             half scale,
                                                             half2& z_prep,
                                                             half2& y_prep) {
  prep_zero_scale_fp16(zero, scale, z_prep, y_prep);
}

// Dequantize one int32 (8 shuffled 4-bit weights) into 4 half2 pairs:
//   dq[0] = (q[0], q[1]) * scale - zero*scale
//   dq[1] = (q[2], q[3]) * scale - zero*scale
//   dq[2] = (q[4], q[5]) * scale - zero*scale
//   dq[3] = (q[6], q[7]) * scale - zero*scale
__forceinline__ __device__ void dequant_4bit_8_fp16(uint32_t qa, half2 (&dq)[4],
                                                    half2 z_prep,
                                                    half2 y_prep) {
  const uint32_t c0 = 0x64006400;

  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = ((qa >> 0) & 0x000F000F) | c0;
  q1.u = ((qa >> 4) & 0x000F000F) | c0;
  q2.u = ((qa >> 8) & 0x000F000F) | c0;
  q3.u = ((qa >> 12) & 0x000F000F) | c0;

  dq[0] = __hmul2(__hsub2(q0.h2, z_prep), y_prep);
  dq[1] = __hmul2(__hsub2(q1.h2, z_prep), y_prep);
  dq[2] = __hmul2(__hsub2(q2.h2, z_prep), y_prep);
  dq[3] = __hmul2(__hsub2(q3.h2, z_prep), y_prep);
}

__forceinline__ __device__ void dequant_4bit_8_fp16_precise(uint32_t qa,
                                                            half2 (&dq)[4],
                                                            half2 z_prep,
                                                            half2 y_prep) {
  dequant_4bit_8_fp16(qa, dq, z_prep, y_prep);
}

// ---------------------------------------------------------------------------
// bf16-input → fp32-output dequant (RDNA3 scalar path).
//
// RDNA3 (gfx1100) has no v_pk_fma_bf16; packed bf16 FMA lowers to a slow
// fallback. Rather than computing dq in bf16 and widening at FMA time in
// the dot product, we widen to fp32 here once (a free left-shift by 16) and
// emit the (q - zero) * scale FMA directly in fp32. This:
//   * Replaces 4× slow bf16 packed FMA with 8× fast fp32 FMA per int32.
//   * Eliminates 4× bf16→fp32 widens that the dot product would do.
//   * Keeps the dot product accumulator in fp32 without a roundtrip.
//
// Output: fp32 dq[8], one element per K position (consumed by the
// fp32-overload of dot22_8_f in q_gemm_rdna3.cu).
__forceinline__ __device__ void prep_zero_scale_bf16_f32(uint32_t zero,
                                                         bf16_t scale,
                                                         float& z_prep,
                                                         float& y_prep) {
  float scale_f = __bfloat162float(scale);
  z_prep = -(128.0f + (float)zero) * scale_f;
  y_prep = scale_f;
}

// Pure-q dequant for the M_COUNT=1 factored path: outputs the unscaled fp32
// values 128+nibble, without folding scale/zero. The caller folds scale/zb
// into the accumulator outside the inner loop using a precomputed sum_a,
// which saves ~27% of the FMA count vs the per-col-dequant approach above
// (only beneficial at M_COUNT=1; break-even at M_COUNT=2).
//
// Cost: 0 FMAs (pure bit-trick + as_float reinterprets).
__forceinline__ __device__ void dequant_4bit_8_bf16_q_only(uint32_t qa,
                                                           float (&q_f32)[8]) {
  const uint32_t c0 = 0x43004300;
  const uint32_t q0 = ((qa >> 0) & 0x000F000F) | c0;
  const uint32_t q1 = ((qa >> 4) & 0x000F000F) | c0;
  const uint32_t q2 = ((qa >> 8) & 0x000F000F) | c0;
  const uint32_t q3 = ((qa >> 12) & 0x000F000F) | c0;
  q_f32[0] = __uint_as_float((q0 & 0xFFFFu) << 16);
  q_f32[1] = __uint_as_float(q0 & 0xFFFF0000u);
  q_f32[2] = __uint_as_float((q1 & 0xFFFFu) << 16);
  q_f32[3] = __uint_as_float(q1 & 0xFFFF0000u);
  q_f32[4] = __uint_as_float((q2 & 0xFFFFu) << 16);
  q_f32[5] = __uint_as_float(q2 & 0xFFFF0000u);
  q_f32[6] = __uint_as_float((q3 & 0xFFFFu) << 16);
  q_f32[7] = __uint_as_float(q3 & 0xFFFF0000u);
}

__forceinline__ __device__ void dequant_4bit_8_bf16_f32(uint32_t qa,
                                                        float (&dq)[8],
                                                        float z_prep,
                                                        float y_prep) {
  const uint32_t c0 = 0x43004300;
  const uint32_t q0 = ((qa >> 0) & 0x000F000F) | c0;
  const uint32_t q1 = ((qa >> 4) & 0x000F000F) | c0;
  const uint32_t q2 = ((qa >> 8) & 0x000F000F) | c0;
  const uint32_t q3 = ((qa >> 12) & 0x000F000F) | c0;
  // bf16(128+nibble) bits → fp32(128+nibble) bits via left-shift by 16
  // (just zero-extends the mantissa from 7 to 23 bits; exponent preserved).
  const float q0x = __uint_as_float((q0 & 0xFFFFu) << 16);
  const float q0y = __uint_as_float(q0 & 0xFFFF0000u);
  const float q1x = __uint_as_float((q1 & 0xFFFFu) << 16);
  const float q1y = __uint_as_float(q1 & 0xFFFF0000u);
  const float q2x = __uint_as_float((q2 & 0xFFFFu) << 16);
  const float q2y = __uint_as_float(q2 & 0xFFFF0000u);
  const float q3x = __uint_as_float((q3 & 0xFFFFu) << 16);
  const float q3y = __uint_as_float(q3 & 0xFFFF0000u);
  // dq[i] = q_f32 * scale + (-(128+zero)*scale) = (nibble - zero) * scale
  dq[0] = __fmaf_rn(q0x, y_prep, z_prep);
  dq[1] = __fmaf_rn(q0y, y_prep, z_prep);
  dq[2] = __fmaf_rn(q1x, y_prep, z_prep);
  dq[3] = __fmaf_rn(q1y, y_prep, z_prep);
  dq[4] = __fmaf_rn(q2x, y_prep, z_prep);
  dq[5] = __fmaf_rn(q2y, y_prep, z_prep);
  dq[6] = __fmaf_rn(q3x, y_prep, z_prep);
  dq[7] = __fmaf_rn(q3y, y_prep, z_prep);
}

}  // namespace gptq_rdna3
}  // namespace vllm

#endif  // _qdq_4_rdna3_cuh
