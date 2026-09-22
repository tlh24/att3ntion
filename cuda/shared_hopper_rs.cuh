#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
// sm_90a R/S backward, D=128, win=32, H%4==0. The caller prepares support[B,N] and
// packed[B,N,ceil(N/32)] (masks already intersected with each query's window).
// Outputs are per-head partials [B,H,N,D]: BF16 if packed_partials (rounded once per
// head after FP32 accumulation), else FP32; the head reduction is the caller's.
extern "C" int att3_shared_rs_wgmma_info(int* info,bool packed_partials=false);
extern "C" int att3_shared_rs_wgmma_w32(
 const __nv_bfloat16* R,const __nv_bfloat16* Vr,
 const __nv_bfloat16* Q,const __nv_bfloat16* dY,
 const __nv_bfloat16* S,const __nv_bfloat16* Vs,
 const float* m,const float* l,const float* delta,
 void* dR,void* dVr,void* dS,void* dVs,
 int B,int H,int N,int win,float scale,cudaStream_t stream,
 const uint8_t* support,const uint32_t* packed,bool packed_partials=false);
