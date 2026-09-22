// Independently derived supporting kernels. No external attention code.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
using bf16 = __nv_bfloat16;

// Four warps handle four independent D128 rows. Multiplication is explicitly
// rounded before summation, matching the original materialized FP32 products.
__global__ void delta128(const bf16* dy, const bf16* y, float* delta, int rows) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  float sum = 0.f;
  if (row < rows) {
    const int64_t base = int64_t(row) * 128;
    #pragma unroll
    for (int d = lane; d < 128; d += 32)
      sum += __fmul_rn(__bfloat162float(dy[base+d]), __bfloat162float(y[base+d]));
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (lane == 0 && row < rows) delta[row] = sum;
}

// Preserve BF16 rounding of each individual head before FP32 head reduction.
// The FP32 reduction order is serial in H, explicitly different from ATen's
// parallel sum tree. This is exact operator semantics, not bitwise equivalence.
template<bool CAST_Q>
__global__ void rounded_heads(const float* q, const float* r, const float* s,
    const float* vr, const float* vs, bf16* dq, bf16* dr, bf16* ds,
    bf16* dvr, bf16* dvs, int H, int ND) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch = blockIdx.y;
  const int tensor = blockIdx.z;
  if (x >= ND) return;
  const float* source = tensor == 0 ? r : tensor == 1 ? s : tensor == 2 ? vr : vs;
  bf16* dest = tensor == 0 ? dr : tensor == 1 ? ds : tensor == 2 ? dvr : dvs;
  float sum = 0.f;
  for (int h = 0; h < H; ++h) {
    const int64_t index = (int64_t(batch) * H + h) * ND + x;
    sum += __bfloat162float(__float2bfloat16_rn(source[index]));
    if constexpr (CAST_Q) {
      if (tensor == 0) dq[index] = __float2bfloat16_rn(q[index]);
    }
  }
  dest[int64_t(batch) * ND + x] = __float2bfloat16_rn(sum);
}

extern "C" int fusion_delta(const bf16* dy, const bf16* y, float* delta,
    int rows, cudaStream_t stream) {
  delta128<<<(rows+3)/4,128,0,stream>>>(dy,y,delta,rows);
  return int(cudaGetLastError());
}

extern "C" int fusion_reduce(int mode, const float* q, const float* r, const float* s,
    const float* vr, const float* vs, bf16* dq, bf16* dr, bf16* ds,
    bf16* dvr, bf16* dvs, int B, int H, int ND, cudaStream_t stream) {
  dim3 grid((ND+255)/256,B,4);
  if (mode == 0)
    rounded_heads<false><<<grid,256,0,stream>>>(q,r,s,vr,vs,dq,dr,ds,dvr,dvs,H,ND);
  else
    rounded_heads<true><<<grid,256,0,stream>>>(q,r,s,vr,vs,dq,dr,ds,dvr,dvs,H,ND);
  return int(cudaGetLastError());
}
