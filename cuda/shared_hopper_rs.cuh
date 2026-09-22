#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
// Four private heads, moving48 projection, phased in-register fold.
// D128,w32,H%4==0; exact masks/support metadata prepared by the caller.
// packed_partials=true writes BF16[B,H,N,D] after full per-head accumulation;
// false writes FP32 partials. Head reduction is external.
extern "C" int att3_shared_rs_wgmma_info(int* info,bool packed_partials=false);
extern "C" int att3_shared_rs_wgmma_w32(
 const __nv_bfloat16* R,const __nv_bfloat16* Vr,
 const __nv_bfloat16* Q,const __nv_bfloat16* dY,
 const __nv_bfloat16* S,const __nv_bfloat16* Vs,
 const float* m,const float* l,const float* delta,
 void* dR,void* dVr,void* dS,void* dVs,
 int B,int H,int N,int win,float scale,cudaStream_t stream,
 const uint8_t* support,const uint32_t* packed,bool packed_partials=false);
