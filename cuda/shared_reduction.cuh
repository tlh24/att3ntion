#include "shared_delta.cuh"

// Partition the head dimension among cooperative thread groups. Each input
// still rounds to BF16 before summation; two short reduction levels replace
// a long dependent head loop. Shared memory is only 1 KiB per CTA.
template<int HG, bool CAST_Q, bool PACKED=false>
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
      if constexpr(PACKED) sum+=__bfloat162float(reinterpret_cast<const bf16*>(source)[index]);
      else sum+=__bfloat162float(__float2bfloat16_rn(source[index]));
      if constexpr(CAST_Q) {
        if(tensor==0)dq[index]=__float2bfloat16_rn(q[index]);
      }
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

extern "C" int fusion_reduce_split(int mode,const float* q,const float* r,const float* s,
    const float* vr,const float* vs,bf16* dq,bf16* dr,bf16* ds,
    bf16* dvr,bf16* dvs,int B,int H,int ND,cudaStream_t stream) {
  #define RUN(ID,HG,CAST) case ID: rounded_heads_split<HG,CAST><<<dim3((ND+256/HG-1)/(256/HG),B,4),256,0,stream>>>(q,r,s,vr,vs,dq,dr,ds,dvr,dvs,H,ND);break;
  switch(mode) {
    RUN(2,2,false) RUN(3,4,false) RUN(4,8,false)
    RUN(5,2,true) RUN(6,4,true) RUN(7,8,true)
    case 8: rounded_heads_split<4,true,true><<<dim3((ND+63)/64,B,4),256,0,stream>>>(q,r,s,vr,vs,dq,dr,ds,dvr,dvs,H,ND);break;
    default:return int(cudaErrorInvalidValue);
  }
  #undef RUN
  return int(cudaGetLastError());
}
