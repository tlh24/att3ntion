#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
// sm_90a R/S backward, D=128, win=64, H%4==0; same contract as shared_hopper_rs.cuh.
// support[B,N] is the visible-key count saturated at 2. Returns a cudaError_t as int.
extern "C" int att3_shared_rs_wgmma64_info(int* info,bool packed_partials=false);
extern "C" int att3_shared_rs_wgmma64_w64(
 const __nv_bfloat16* R,const __nv_bfloat16* Vr,const __nv_bfloat16* Q,const __nv_bfloat16* dY,const __nv_bfloat16* S,const __nv_bfloat16* Vs,
 const float* m,const float* l,const float* delta,void* dR,void* dVr,void* dS,void* dVs,
 int B,int H,int N,int win,float scale,cudaStream_t stream,const uint8_t* support,const uint32_t* packed,bool packed_partials=false);
