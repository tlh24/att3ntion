#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace att3_mask_metadata {
// One warp owns a query row. Metadata is shared by every head and gradient
// direction. Saturated support is sufficient for exact empty/singleton rules.
__global__ void prepare(const bool* mask,uint32_t* words,uint8_t* support,
    int B,int N,int win,int row_words) {
  const int lane=threadIdx.x&31;
  const int row=blockIdx.x*4+(threadIdx.x>>5);
  const bool live=row<B*N;
  const int query=row%N;
  const int lo=max(0,query-win+1);
  unsigned count=0;
  for(int word=0;word<row_words;++word) {
    const int key=word*32+lane;
    const bool visible=live && key>=lo && key<=query && key<N;
    const uint32_t bits=__ballot_sync(0xffffffff,visible && mask[int64_t(row)*N+key]);
    if(lane==0 && live) {
      words[int64_t(row)*row_words+word]=bits;
      count=min(2u,count+unsigned(__popc(bits)));
    }
  }
  if(lane==0 && live)support[row]=uint8_t(count);
}
inline cudaError_t launch(const bool* mask,uint32_t* words,uint8_t* support,
    int B,int N,int win,cudaStream_t stream) {
  prepare<<<(B*N+3)/4,128,0,stream>>>(mask,words,support,B,N,win,(N+31)/32);
  return cudaGetLastError();
}
} // namespace att3_mask_metadata
