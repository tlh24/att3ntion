// Shared-KV backward helper: delta = rowsum(dY * Y).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
using bf16 = __nv_bfloat16;

// One warp per D=128 row, four rows per CTA. __fmul_rn keeps each product
// rounded before the sum (no FMA), as if dY * Y were materialized in fp32.
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

extern "C" int fusion_delta(const bf16* dy, const bf16* y, float* delta,
    int rows, cudaStream_t stream) {
  delta128<<<(rows+3)/4,128,0,stream>>>(dy,y,delta,rows);
  return int(cudaGetLastError());
}
