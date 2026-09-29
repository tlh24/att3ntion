#include "shared_delta.cuh"

// Head reduction with H split across HG thread groups, then summed in shared
// memory. The sources hold bf16 per-head partials; dQ is cast from fp32.
template<int HG>
__global__ void rounded_heads_split(const float* q, const float* r, const float* s,
    const float* vr, const float* vs, bf16* dq, bf16* dr, bf16* ds,
    bf16* dvr, bf16* dvs, int H, int ND) {
  constexpr int CHANNELS=256/HG;
  __shared__ float partial[HG][CHANNELS];
  const int hx=threadIdx.x/CHANNELS,cx=threadIdx.x%CHANNELS;
  const int x=blockIdx.x*CHANNELS+cx,batch=blockIdx.y,tensor=blockIdx.z;
  const float* source=tensor==0?r:tensor==1?s:tensor==2?vr:vs;
  bf16* dest=tensor==0?dr:tensor==1?ds:tensor==2?dvr:dvs;
  float sum=0.f;
  if(x<ND) {
    for(int h=hx;h<H;h+=HG) {
      const int64_t index=(int64_t(batch)*H+h)*ND+x;
      sum+=__bfloat162float(reinterpret_cast<const bf16*>(source)[index]);
      if(tensor==0)dq[index]=__float2bfloat16_rn(q[index]);
    }
  }
  partial[hx][cx]=sum;
  __syncthreads();
  if(hx==0 && x<ND) {
    float total=partial[0][cx];
    #pragma unroll
    for(int h=1;h<HG;++h)total+=partial[h][cx];
    dest[int64_t(batch)*ND+x]=__float2bfloat16_rn(total);
  }
}

extern "C" int fusion_reduce_split(const float* q,const float* r,const float* s,
    const float* vr,const float* vs,bf16* dq,bf16* dr,bf16* ds,
    bf16* dvr,bf16* dvs,int B,int H,int ND,cudaStream_t stream) {
  rounded_heads_split<4><<<dim3((ND+63)/64,B,4),256,0,stream>>>(q,r,s,vr,vs,dq,dr,ds,dvr,dvs,H,ND);
  return int(cudaGetLastError());
}
