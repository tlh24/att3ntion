/**
 * @file common.cuh
 * @brief Shared constants, utilities, and kernels for hypergraph attention.
 *
 * This header consolidates common definitions used by both forward and backward
 * CUDA kernels, eliminating duplication and ensuring consistency.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// =============================================================================
// Numerical Stability Constants
// =============================================================================

/** Maximum exponent value to prevent overflow in expf() */
constexpr float EXP_CLIP = 80.0f;

/** Epsilon to prevent division by zero in softmax denominators */
constexpr float DENOM_EPS = 1e-6f;

/** Large negative value for masked/padded attention scores */
constexpr float NEG_INF = -1e30f;

// =============================================================================
// Tile Size Configuration
// =============================================================================
// These control shared memory usage and parallelism. Tune based on GPU arch.

#ifndef TILE_I
#define TILE_I 16
#endif

#ifndef TILE_J
#define TILE_J 16
#endif

#ifndef TILE_K
#define TILE_K 16
#endif

// Smaller tiles for scatter kernels (higher register pressure)
#ifndef TILE_I_SCATTER
#define TILE_I_SCATTER 4
#endif

#ifndef TILE_J_SCATTER
#define TILE_J_SCATTER 4
#endif

#ifndef TILE_K_SCATTER
#define TILE_K_SCATTER 4
#endif

// Maximum embedding dimension that fits in registers
#ifndef MAX_D_REG
#define MAX_D_REG 64
#endif

// =============================================================================
// Utility Functions
// =============================================================================

/** Integer division rounded up */
__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

/** Three-way element-wise dot product: sum(a[d] * b[d] * c[d]) */
__device__ __forceinline__ float dot3(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int D
) {
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        sum += a[d] * b[d] * c[d];
    }
    return sum;
}

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf2f(bf16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ bf16 f2bf(float x) {
    return __float2bfloat16(x);
}

