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

// =============================================================================
// Tensor-core primitives (sm_80+), shared by Y_gather_tc and Bwd_gather_tc
// =============================================================================

__device__ __forceinline__ void mma_bf16_m16n8k16(
    float c[4], const uint32_t a[4], const uint32_t b[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ uint32_t pack_bf162(float x, float y) {
    __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const bf16* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const bf16* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}

__device__ __forceinline__ void cp_async16(bf16* dst, const bf16* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
#endif
}

