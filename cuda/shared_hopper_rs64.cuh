#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
// Optional sm_90a TU. Exact windows/holes require support[B,N] saturated0/1/2+
// and packed[B,N,ceil(N/32)] to already intersect each query's causal window.
// Return is cudaError_t encoded as int. Supported: D128,w64,H divisible4.
// No metadata preparation or final head reduction is hidden in this launcher.
extern "C" int att3_shared_rs_wgmma64_info(int* info,bool packed_partials=false);
extern "C" int att3_shared_rs_wgmma64_w64(
 const __nv_bfloat16* R,const __nv_bfloat16* Vr,const __nv_bfloat16* Q,const __nv_bfloat16* dY,const __nv_bfloat16* S,const __nv_bfloat16* Vs,
 const float* m,const float* l,const float* delta,void* dR,void* dVr,void* dS,void* dVs,
 int B,int H,int N,int win,float scale,cudaStream_t stream,const uint8_t* support,const uint32_t* packed,bool packed_partials=false);
// packed_partials=true: outputs point to BF16[B,H,N,128], rounded only
// after each complete per-head FP32 contraction. false: original FP32 outputs.
