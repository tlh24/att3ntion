// sm_90a WGMMA forward for the shared-KV single gather (D=128, Hkv=1). One CTA per
// query and head group: the query's R/Vr/S/Vs window is staged once and reused by
// every head the CTA visits. Built as a separate object with CuTe (see setup.py).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
namespace att3_shared_hopper_impl {
using namespace cute;
using bf16 = cute::bfloat16_t;
constexpr int D=128;
constexpr float NEG=-1e30f;
template<int N>struct Atom;
template<>struct Atom<16>{using T=SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<64>{using T=SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<32>{using T=SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<128>{using T=SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>;};
template<int W>struct PSwizzle;
template<>struct PSwizzle<16>{using T=GMMA::Layout_K_SW32_Atom<bf16>;};
template<>struct PSwizzle<64>{using T=GMMA::Layout_K_SW128_Atom<bf16>;};
template<>struct PSwizzle<32>{using T=GMMA::Layout_K_SW64_Atom<bf16>;};
// Check the hand-written accumulator (row, col) indexing against CuTe's layout.
template<int N,int Tid>constexpr bool mapping_valid(){
 using Mma=decltype(make_tiled_mma(typename Atom<N>::T{}));
 using TV=decltype(Mma{}.get_layoutC_TV());constexpr TV tv{};
 if(size<0>(tv)!=128 || size<1>(tv)!=N/2)return false;
 for(int nt=0;nt<N/8;++nt)for(int e=0;e<4;++e){
  const int coord=tv(make_coord(Tid,4*nt+e));
  const int row=(Tid/32)*16+(Tid%32)/4+(e/2)*8;
  const int col=nt*8+2*(Tid%4)+(e%2);
  if(coord%64!=row || coord/64!=col)return false;
 }
 return true;
}
template<int N,int Tid>struct VerifiedMapping{static_assert(mapping_valid<N,Tid>(),"Re-derive register ownership for changed CuTe atom");static constexpr bool value=true;};
template<int N,int...T>constexpr bool all_mappings(std::integer_sequence<int,T...>){return (VerifiedMapping<N,T>::value&&...);}
static_assert(all_mappings<16>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<64>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<32>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<128>(std::make_integer_sequence<int,128>{}));
__device__ __forceinline__ void copy16(void* sm,const void* gm){
 unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(sm));
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(gm));
}

// Window W, G=64/W heads per warpgroup with W/16 warps each: the 64 MMA rows are
// G heads x W keys j. HV head groups are visited serially.
template<int W,int HV=1>
__global__ __launch_bounds__(128) void retained_warpgroup(
 const bf16* __restrict__ Q,const bf16* __restrict__ R,const bf16* __restrict__ S,
 const bf16* __restrict__ Vr,const bf16* __restrict__ Vs,bf16* __restrict__ Y,
 float* __restrict__ mout,float* __restrict__ lout,const bool* __restrict__ mask,
 int H,int N,float scale,int win){
 constexpr int G=64/W,WPH=W/16,NT=D/8,CT=W/8;
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane/4,tig=lane%4;
 const int lh=warp/WPH,th=tid%(WPH*32),i=blockIdx.x,group_h0=blockIdx.y*G*HV,b=blockIdx.z;
 const int lo=max(0,i-W+1),hi=min(N,i+1);
 const int row0=warp*16+g,row1=row0+8;
 const int j0=lo+row0%W,j1=lo+row1%W;
 const int64_t kv=(int64_t)b*N*D;
 const bool* mrow=mask+((int64_t)b*N+i)*N;
 using Score=decltype(make_tiled_mma(typename Atom<W>::T{}));
 using Value=decltype(make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{}));
 auto al=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<64>{},Int<D>{}));
 auto sl=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<W>{},Int<D>{}));
 auto pl=tile_to_shape(typename PSwizzle<W>::T{},make_shape(Int<64>{},Int<W>{}));
 auto vl=tile_to_shape(GMMA::Layout_MN_SW128_Atom<bf16>{},make_shape(Int<D>{},Int<W>{}));
 extern __shared__ __align__(128) char storage[];
 bf16* ap=reinterpret_cast<bf16*>(storage);
 bf16* sp=ap+cosize_v<decltype(al)>;
 bf16* pp=sp+cosize_v<decltype(sl)>;
 bf16* vp=pp+cosize_v<decltype(pl)>;
 bf16* rawr=vp+cosize_v<decltype(vl)>;
 bf16* rawvr=rawr+W*D;
 float* aq=reinterpret_cast<float*>(rawvr+W*(D+8));
 float* wn=aq+G*D;
 float* wml=wn+4*D;
 auto A=make_tensor(make_smem_ptr(ap),al);
 auto B=make_tensor(make_smem_ptr(sp),sl);
 auto P=make_tensor(make_smem_ptr(pp),pl);
 auto V=make_tensor(make_smem_ptr(vp),vl);
 for(int idx=tid;idx<W*(D/8);idx+=128){
  const int j=idx/(D/8),d=idx%(D/8)*8,tok=lo+j;
  if(tok<N){
   const int64_t off=kv+(int64_t)tok*D+d;
   copy16(rawr+j*D+d,R+off);copy16(rawvr+j*(D+8)+d,Vr+off);
   copy16(&B(j,d),S+off);copy16(&V(d,j),Vs+off);
  }else{
   uint4 z=make_uint4(0,0,0,0);
   *reinterpret_cast<uint4*>(rawr+j*D+d)=z;*reinterpret_cast<uint4*>(rawvr+j*(D+8)+d)=z;
   *reinterpret_cast<uint4*>(&B(j,d))=z;*reinterpret_cast<uint4*>(&V(d,j))=z;
  }
 }
 #pragma unroll 1
 for(int visit=0;visit<HV;++visit){
 const int h0=group_h0+visit*G;
 for(int idx=tid;idx<G*D;idx+=128){const int head=idx/D,d=idx%D;aq[idx]=scale*float(Q[(((int64_t)b*H+h0+head)*N+i)*D+d]);}
 asm volatile("cp.async.wait_all;"::);
 __syncthreads();

 // each thread visits different rows but the same eight channels: load its
 // scaled-Q fragment once, scoped so the registers die before the epilogue
 {
 const int fragment_d=(th%(D/8))*8;
 float anchor_fragment[8];
 #pragma unroll
 for(int e=0;e<8;++e)anchor_fragment[e]=aq[lh*D+fragment_d+e];
 for(int idx=th;idx<W*(D/8);idx+=WPH*32){
  const int row=idx/(D/8),d=idx%(D/8)*8;
  uint4 v=*reinterpret_cast<const uint4*>(rawr+row*D+d);
  auto* pairs=reinterpret_cast<__nv_bfloat162*>(&v);
  #pragma unroll
  for(int e=0;e<4;++e){auto f=__bfloat1622float2(pairs[e]);pairs[e]=__floats2bfloat162_rn(anchor_fragment[e*2]*f.x,anchor_fragment[e*2+1]*f.y);}
  *reinterpret_cast<uint4*>(&A(lh*W+row,d))=v;
 }
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 Score smma;auto st=smma.get_thread_slice(tid);
 auto fa=st.make_fragment_A(st.partition_A(A));auto fb=st.make_fragment_B(st.partition_B(B));
 auto cx=make_identity_tensor(make_shape(Int<64>{},Int<W>{}));auto coord=st.partition_C(cx);auto x=st.make_fragment_C(coord);clear(x);
 warpgroup_fence_operand(x);warpgroup_arrive();gemm(smma,fa,fb,x);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(x);
 float m0=NEG,m1=NEG;
 const bool rm0=j0<hi&&mrow[j0],rm1=j1<hi&&mrow[j1];
 #pragma unroll
 for(int nt=0;nt<CT;++nt){
  const int c0=lo+nt*8+2*tig,c1=c0+1;
  const bool cm0=c0<hi&&mrow[c0],cm1=c1<hi&&mrow[c1];
  x(4*nt)=rm0&&cm0?x(4*nt):NEG;x(4*nt+1)=rm0&&cm1?x(4*nt+1):NEG;
  x(4*nt+2)=rm1&&cm0?x(4*nt+2):NEG;x(4*nt+3)=rm1&&cm1?x(4*nt+3):NEG;
  m0=fmaxf(m0,fmaxf(x(4*nt),x(4*nt+1)));m1=fmaxf(m1,fmaxf(x(4*nt+2),x(4*nt+3)));
 }
 #pragma unroll
 for(int off=1;off<=2;off<<=1){m0=fmaxf(m0,__shfl_xor_sync(0xffffffff,m0,off));m1=fmaxf(m1,__shfl_xor_sync(0xffffffff,m1,off));}
 float l0=0,l1=0;
 #pragma unroll
 for(int nt=0;nt<CT;++nt){
  const int c=nt*8+2*tig;
  const float p0=x(4*nt)<-5e29f?0.f:__expf(x(4*nt)-m0),p1=x(4*nt+1)<-5e29f?0.f:__expf(x(4*nt+1)-m0);
  const float p2=x(4*nt+2)<-5e29f?0.f:__expf(x(4*nt+2)-m1),p3=x(4*nt+3)<-5e29f?0.f:__expf(x(4*nt+3)-m1);
  *reinterpret_cast<__nv_bfloat162*>(&P(row0,c))=__floats2bfloat162_rn(p0,p1);
  *reinterpret_cast<__nv_bfloat162*>(&P(row1,c))=__floats2bfloat162_rn(p2,p3);
  l0+=p0+p1;l1+=p2+p3;
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 Value vmma;auto vt=vmma.get_thread_slice(tid);
 auto fp=vt.make_fragment_A(vt.partition_A(P));auto fv=vt.make_fragment_B(vt.partition_B(V));
 auto cu=make_identity_tensor(make_shape(Int<64>{},Int<D>{}));auto uc=vt.partition_C(cu);auto u=vt.make_fragment_C(uc);clear(u);
 warpgroup_fence_operand(u);warpgroup_arrive();gemm(vmma,fp,fv,u);warpgroup_commit_batch();
 // scalar normalization overlaps the asynchronous value MMA
 #pragma unroll
 for(int off=1;off<=2;off<<=1){l0+=__shfl_xor_sync(0xffffffff,l0,off);l1+=__shfl_xor_sync(0xffffffff,l1,off);}
 float mw=fmaxf(m0,m1);
 #pragma unroll
 for(int off=16;off>0;off>>=1)mw=fmaxf(mw,__shfl_xor_sync(0xffffffff,mw,off));
 const float w0=__expf(m0-mw),w1=__expf(m1-mw);
 float lw=w0*l0+w1*l1;
 #pragma unroll
 for(int off=4;off<=16;off<<=1)lw+=__shfl_xor_sync(0xffffffff,lw,off);
 if(lane==0){wml[warp*2]=mw;wml[warp*2+1]=lw;}
 warpgroup_wait<0>();warpgroup_fence_operand(u);
 float acc[2*NT];
 #pragma unroll
 for(int nt=0;nt<NT;++nt){
  const int c=nt*8+2*tig;
  const auto v0=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawvr+(row0%W)*(D+8)+c));
  const auto v1=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawvr+(row1%W)*(D+8)+c));
  acc[nt*2]=w0*v0.x*u(4*nt)+w1*v1.x*u(4*nt+2);
  acc[nt*2+1]=w0*v0.y*u(4*nt+1)+w1*v1.y*u(4*nt+3);
 }
 #pragma unroll
 for(int off=4;off<=16;off<<=1){
  #pragma unroll
  for(int e=0;e<2*NT;++e)acc[e]+=__shfl_xor_sync(0xffffffff,acc[e],off);
 }
 if(lane<4){
  #pragma unroll
  for(int nt=0;nt<NT;++nt){wn[warp*D+nt*8+2*lane]=acc[2*nt];wn[warp*D+nt*8+2*lane+1]=acc[2*nt+1];}
 }
 __syncthreads();
 float mh=NEG;
 #pragma unroll
 for(int w=0;w<WPH;++w)mh=fmaxf(mh,wml[(lh*WPH+w)*2]);
 float l=0;
 #pragma unroll
 for(int w=0;w<WPH;++w)l+=__expf(wml[(lh*WPH+w)*2]-mh)*wml[(lh*WPH+w)*2+1];
 const float inv=l>1e-20f?1.f/l:0.f;const int64_t out=((int64_t)b*H+h0+lh)*N+i;
 for(int d=th;d<D;d+=WPH*32){float val=0;
  #pragma unroll
  for(int w=0;w<WPH;++w)val+=__expf(wml[(lh*WPH+w)*2]-mh)*wn[(lh*WPH+w)*D+d];
  Y[out*D+d]=bf16(val*inv);
 }
 if(th==0){mout[out]=mh;lout[out]=l;}
 } // head-group visit
}

template<int W>constexpr size_t bytes(){return sizeof(bf16)*(64*D+W*D+64*W+D*W+2*W*D+8*W)+sizeof(float)*((64/W)*D+4*D+8);}

template<int W,int HV>bool launch(const void* q,const void* r,const void* s,const void* vr,const void* vs,
 void* y,float* m,float* l,const bool* mask,int B,int H,int N,float scale,int max_smem,cudaStream_t stream,int device){
 constexpr size_t smem=bytes<W>();
 if(smem>(size_t)max_smem)return false;
 static thread_local int attribute_device=-1;
 if(attribute_device!=device){
  auto e=cudaFuncSetAttribute(retained_warpgroup<W,HV>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(e!=cudaSuccess)return false;
  attribute_device=device;
 }
 retained_warpgroup<W,HV><<<dim3(N,H/((64/W)*HV),B),128,smem,stream>>>(
  reinterpret_cast<const bf16*>(q),reinterpret_cast<const bf16*>(r),reinterpret_cast<const bf16*>(s),
  reinterpret_cast<const bf16*>(vr),reinterpret_cast<const bf16*>(vs),reinterpret_cast<bf16*>(y),m,l,mask,H,N,scale,W);
 return true;
}
}
namespace att3_shared_hopper_split_impl {
using namespace cute;
using bf16 = cute::bfloat16_t;
constexpr int D=128;
constexpr float NEG=-1e30f;
template<int N>struct Atom;
template<>struct Atom<16>{using T=SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<64>{using T=SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<32>{using T=SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<128>{using T=SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>;};
template<int W>struct PSwizzle;
template<>struct PSwizzle<16>{using T=GMMA::Layout_K_SW32_Atom<bf16>;};
template<>struct PSwizzle<64>{using T=GMMA::Layout_K_SW128_Atom<bf16>;};
template<>struct PSwizzle<32>{using T=GMMA::Layout_K_SW64_Atom<bf16>;};
template<int N,int Tid>constexpr bool mapping_valid(){
 using Mma=decltype(make_tiled_mma(typename Atom<N>::T{}));
 using TV=decltype(Mma{}.get_layoutC_TV());constexpr TV tv{};
 if(size<0>(tv)!=128 || size<1>(tv)!=N/2)return false;
 for(int nt=0;nt<N/8;++nt)for(int e=0;e<4;++e){
  const int coord=tv(make_coord(Tid,4*nt+e));
  const int row=(Tid/32)*16+(Tid%32)/4+(e/2)*8;
  const int col=nt*8+2*(Tid%4)+(e%2);
  if(coord%64!=row || coord/64!=col)return false;
 }
 return true;
}
template<int N,int Tid>struct VerifiedMapping{static_assert(mapping_valid<N,Tid>(),"Re-derive register ownership for changed CuTe atom");static constexpr bool value=true;};
template<int N,int...T>constexpr bool all_mappings(std::integer_sequence<int,T...>){return (VerifiedMapping<N,T>::value&&...);}
static_assert(all_mappings<16>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<64>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<32>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<128>(std::make_integer_sequence<int,128>{}));
__device__ __forceinline__ void copy16(void* sm,const void* gm){
 unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(sm));
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(gm));
}

// retained_warpgroup with the value MMA split into two 64-channel halves, halving the
// value accumulator and operand. STASH holds Vs in registers across the score MMA.
template<int W,int HV=1,bool STASH=false>
__global__ __launch_bounds__(128) void split_value_warpgroup(
 const bf16* __restrict__ Q,const bf16* __restrict__ R,const bf16* __restrict__ S,
 const bf16* __restrict__ Vr,const bf16* __restrict__ Vs,bf16* __restrict__ Y,
 float* __restrict__ mout,float* __restrict__ lout,const bool* __restrict__ mask,
 int H,int N,float scale,int win){
 constexpr int G=64/W,WPH=W/16,VD=64,NT=VD/8,CT=W/8;
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane/4,tig=lane%4;
 const int lh=warp/WPH,th=tid%(WPH*32),i=blockIdx.x,group_h0=blockIdx.y*G*HV,b=blockIdx.z;
 const int lo=max(0,i-W+1),hi=min(N,i+1);
 const int row0=warp*16+g,row1=row0+8;
 const int j0=lo+row0%W,j1=lo+row1%W;
 const int64_t kv=(int64_t)b*N*D;
 const bool* mrow=mask+((int64_t)b*N+i)*N;
 using Score=decltype(make_tiled_mma(typename Atom<W>::T{}));
 using Value=decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{}));
 auto al=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<64>{},Int<D>{}));
 auto sl=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<W>{},Int<D>{}));
 auto pl=tile_to_shape(typename PSwizzle<W>::T{},make_shape(Int<64>{},Int<W>{}));
 auto vl=tile_to_shape(GMMA::Layout_MN_SW128_Atom<bf16>{},make_shape(Int<VD>{},Int<W>{}));
 extern __shared__ __align__(128) char storage[];
 bf16* ap=reinterpret_cast<bf16*>(storage);
 bf16* sp=ap+cosize_v<decltype(al)>;
 bf16* pp=ap; // score operand is dead before probability writes
 bf16* vp=sp+cosize_v<decltype(sl)>;
 bf16* rawr=vp+cosize_v<decltype(vl)>;
 bf16* rawvr=rawr+W*D;
 float* aq=reinterpret_cast<float*>(rawvr+W*(D+8));
 float* wn=reinterpret_cast<float*>(ap+64*W); // upper dead A half, disjoint from P
 float* wml=aq+G*D;
 auto A=make_tensor(make_smem_ptr(ap),al);
 auto B=make_tensor(make_smem_ptr(sp),sl);
 auto P=make_tensor(make_smem_ptr(pp),pl);
 auto V=make_tensor(make_smem_ptr(vp),vl);
 uint4 vstash[STASH ? 2*(W*(VD/8)/128) : 1];
 if constexpr(STASH){
  #pragma unroll
  for(int part=0;part<2;++part){
   #pragma unroll
   for(int k=0;k<W*(VD/8)/128;++k){
    const int idx=tid+k*128,j=idx/(VD/8),d=idx%(VD/8)*8+part*VD,tok=lo+j;
    vstash[part*(W*(VD/8)/128)+k]=tok<N?*reinterpret_cast<const uint4*>(Vs+kv+(int64_t)tok*D+d):make_uint4(0,0,0,0);
   }
  }
 }
 for(int idx=tid;idx<W*(D/8);idx+=128){
  const int j=idx/(D/8),d=idx%(D/8)*8,tok=lo+j;
  if(tok<N){
   const int64_t off=kv+(int64_t)tok*D+d;
   copy16(rawr+j*D+d,R+off);copy16(rawvr+j*(D+8)+d,Vr+off);
   copy16(&B(j,d),S+off);
  }else{
   uint4 z=make_uint4(0,0,0,0);
   *reinterpret_cast<uint4*>(rawr+j*D+d)=z;*reinterpret_cast<uint4*>(rawvr+j*(D+8)+d)=z;
   *reinterpret_cast<uint4*>(&B(j,d))=z;
  }
 }
 #pragma unroll 1
 for(int visit=0;visit<HV;++visit){
 const int h0=group_h0+visit*G;
 for(int idx=tid;idx<G*D;idx+=128){const int head=idx/D,d=idx%D;aq[idx]=scale*float(Q[(((int64_t)b*H+h0+head)*N+i)*D+d]);}
 asm volatile("cp.async.wait_all;"::);
 __syncthreads();
 for(int idx=th;idx<W*(D/8);idx+=WPH*32){
  const int row=idx/(D/8),d=idx%(D/8)*8;
  uint4 v=*reinterpret_cast<const uint4*>(rawr+row*D+d);
  auto* pairs=reinterpret_cast<__nv_bfloat162*>(&v);
  #pragma unroll
  for(int e=0;e<4;++e){auto f=__bfloat1622float2(pairs[e]);pairs[e]=__floats2bfloat162_rn(aq[lh*D+d+e*2]*f.x,aq[lh*D+d+e*2+1]*f.y);}
  *reinterpret_cast<uint4*>(&A(lh*W+row,d))=v;
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 Score smma;auto st=smma.get_thread_slice(tid);
 auto fa=st.make_fragment_A(st.partition_A(A));auto fb=st.make_fragment_B(st.partition_B(B));
 auto cx=make_identity_tensor(make_shape(Int<64>{},Int<W>{}));auto coord=st.partition_C(cx);auto x=st.make_fragment_C(coord);clear(x);
 warpgroup_fence_operand(x);warpgroup_arrive();gemm(smma,fa,fb,x);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(x);
 float m0=NEG,m1=NEG;
 const bool rm0=j0<hi&&mrow[j0],rm1=j1<hi&&mrow[j1];
 #pragma unroll
 for(int nt=0;nt<CT;++nt){
  const int c0=lo+nt*8+2*tig,c1=c0+1;
  const bool cm0=c0<hi&&mrow[c0],cm1=c1<hi&&mrow[c1];
  x(4*nt)=rm0&&cm0?x(4*nt):NEG;x(4*nt+1)=rm0&&cm1?x(4*nt+1):NEG;
  x(4*nt+2)=rm1&&cm0?x(4*nt+2):NEG;x(4*nt+3)=rm1&&cm1?x(4*nt+3):NEG;
  m0=fmaxf(m0,fmaxf(x(4*nt),x(4*nt+1)));m1=fmaxf(m1,fmaxf(x(4*nt+2),x(4*nt+3)));
 }
 #pragma unroll
 for(int off=1;off<=2;off<<=1){m0=fmaxf(m0,__shfl_xor_sync(0xffffffff,m0,off));m1=fmaxf(m1,__shfl_xor_sync(0xffffffff,m1,off));}
 float l0=0,l1=0;
 #pragma unroll
 for(int nt=0;nt<CT;++nt){
  const int c=nt*8+2*tig;
  const float p0=x(4*nt)<-5e29f?0.f:__expf(x(4*nt)-m0),p1=x(4*nt+1)<-5e29f?0.f:__expf(x(4*nt+1)-m0);
  const float p2=x(4*nt+2)<-5e29f?0.f:__expf(x(4*nt+2)-m1),p3=x(4*nt+3)<-5e29f?0.f:__expf(x(4*nt+3)-m1);
  *reinterpret_cast<__nv_bfloat162*>(&P(row0,c))=__floats2bfloat162_rn(p0,p1);
  *reinterpret_cast<__nv_bfloat162*>(&P(row1,c))=__floats2bfloat162_rn(p2,p3);
  l0+=p0+p1;l1+=p2+p3;
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 #pragma unroll
 for(int off=1;off<=2;off<<=1){l0+=__shfl_xor_sync(0xffffffff,l0,off);l1+=__shfl_xor_sync(0xffffffff,l1,off);}
 float mw=fmaxf(m0,m1);
 #pragma unroll
 for(int off=16;off>0;off>>=1)mw=fmaxf(mw,__shfl_xor_sync(0xffffffff,mw,off));
 const float w0=__expf(m0-mw),w1=__expf(m1-mw);
 float lw=w0*l0+w1*l1;
 #pragma unroll
 for(int off=4;off<=16;off<<=1)lw+=__shfl_xor_sync(0xffffffff,lw,off);
 if(lane==0){wml[warp*2]=mw;wml[warp*2+1]=lw;}

 #pragma unroll
 for(int part=0;part<2;++part){
  #pragma unroll
  for(int k=0;k<W*(VD/8)/128;++k){
   const int idx=tid+k*128,j=idx/(VD/8),d=idx%(VD/8)*8,tok=lo+j;
   if constexpr(STASH){*reinterpret_cast<uint4*>(&V(d,j))=vstash[part*(W*(VD/8)/128)+k];}
   else if(tok<N){copy16(&V(d,j),Vs+kv+(int64_t)tok*D+part*VD+d);}
   else{*reinterpret_cast<uint4*>(&V(d,j))=make_uint4(0,0,0,0);}
  }
  asm volatile("cp.async.wait_all;"::);
  cutlass::arch::fence_view_async_shared();__syncthreads();
  Value vmma;auto vt=vmma.get_thread_slice(tid);
  auto fp=vt.make_fragment_A(vt.partition_A(P));auto fv=vt.make_fragment_B(vt.partition_B(V));
  auto cu=make_identity_tensor(make_shape(Int<64>{},Int<VD>{}));auto uc=vt.partition_C(cu);auto u=vt.make_fragment_C(uc);clear(u);
  warpgroup_fence_operand(u);warpgroup_arrive();gemm(vmma,fp,fv,u);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(u);
 float acc[2*NT];
 #pragma unroll
 for(int nt=0;nt<NT;++nt){
  const int c=part*VD+nt*8+2*tig;
  const auto v0=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawvr+(row0%W)*(D+8)+c));
  const auto v1=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawvr+(row1%W)*(D+8)+c));
  acc[nt*2]=w0*v0.x*u(4*nt)+w1*v1.x*u(4*nt+2);
  acc[nt*2+1]=w0*v0.y*u(4*nt+1)+w1*v1.y*u(4*nt+3);
 }
 #pragma unroll
 for(int off=4;off<=16;off<<=1){
  #pragma unroll
  for(int e=0;e<2*NT;++e)acc[e]+=__shfl_xor_sync(0xffffffff,acc[e],off);
 }
 if(lane<4){
  #pragma unroll
  for(int nt=0;nt<NT;++nt){wn[warp*VD+nt*8+2*lane]=acc[2*nt];wn[warp*VD+nt*8+2*lane+1]=acc[2*nt+1];}
 }
 __syncthreads();
 float mh=NEG;
 #pragma unroll
 for(int w=0;w<WPH;++w)mh=fmaxf(mh,wml[(lh*WPH+w)*2]);
 float l=0;
 #pragma unroll
 for(int w=0;w<WPH;++w)l+=__expf(wml[(lh*WPH+w)*2]-mh)*wml[(lh*WPH+w)*2+1];
 const float inv=l>1e-20f?1.f/l:0.f;const int64_t out=((int64_t)b*H+h0+lh)*N+i;
 for(int d=th;d<VD;d+=WPH*32){float val=0;
  #pragma unroll
  for(int w=0;w<WPH;++w)val+=__expf(wml[(lh*WPH+w)*2]-mh)*wn[(lh*WPH+w)*VD+d];
  Y[out*D+part*VD+d]=bf16(val*inv);
 }
 if(th==0){mout[out]=mh;lout[out]=l;}

  __syncthreads(); // V and wn are reused by the next channel half
 } // channel half
 } // head-group visit
}

template<int W>constexpr size_t bytes(){return sizeof(bf16)*(64*D+W*D+(D/2)*W+2*W*D+8*W)+sizeof(float)*((64/W)*D+8);}
template<int W,int HV>bool launch(const void* q,const void* r,const void* s,const void* vr,const void* vs,
 void* y,float* m,float* l,const bool* mask,int B,int H,int N,float scale,int max_smem,cudaStream_t stream,int device){
 constexpr size_t smem=bytes<W>();
 if(smem>(size_t)max_smem)return false;
 static thread_local int attribute_device=-1;
 if(attribute_device!=device){
  auto e=cudaFuncSetAttribute(split_value_warpgroup<W,HV,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(e!=cudaSuccess)return false;
  attribute_device=device;
 }
 split_value_warpgroup<W,HV,true><<<dim3(N,H/((64/W)*HV),B),128,smem,stream>>>(
  reinterpret_cast<const bf16*>(q),reinterpret_cast<const bf16*>(r),reinterpret_cast<const bf16*>(s),
  reinterpret_cast<const bf16*>(vr),reinterpret_cast<const bf16*>(vs),reinterpret_cast<bf16*>(y),m,l,mask,H,N,scale,W);
 return true;
}
}

namespace att3_shared_hopper_w128_impl {
using namespace cute;
using bf16 = cute::bfloat16_t;
constexpr int D=128;
constexpr float NEG=-1e30f;
template<int N>struct Atom;
template<>struct Atom<16>{using T=SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<64>{using T=SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<32>{using T=SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct Atom<128>{using T=SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;};
template<int W>struct PSwizzle;
template<>struct PSwizzle<128>{using T=GMMA::Layout_K_SW128_Atom<bf16>;};
template<>struct PSwizzle<16>{using T=GMMA::Layout_K_SW32_Atom<bf16>;};
template<>struct PSwizzle<64>{using T=GMMA::Layout_K_SW128_Atom<bf16>;};
template<>struct PSwizzle<32>{using T=GMMA::Layout_K_SW64_Atom<bf16>;};
template<int N,int Tid>constexpr bool mapping_valid(){
 using Mma=decltype(make_tiled_mma(typename Atom<N>::T{}));
 using TV=decltype(Mma{}.get_layoutC_TV());constexpr TV tv{};
 if(size<0>(tv)!=128 || size<1>(tv)!=N/2)return false;
 for(int nt=0;nt<N/8;++nt)for(int e=0;e<4;++e){
  const int coord=tv(make_coord(Tid,4*nt+e));
  const int row=(Tid/32)*16+(Tid%32)/4+(e/2)*8;
  const int col=nt*8+2*(Tid%4)+(e%2);
  if(coord%64!=row || coord/64!=col)return false;
 }
 return true;
}
template<int N,int Tid>struct VerifiedMapping{static_assert(mapping_valid<N,Tid>(),"Re-derive register ownership for changed CuTe atom");static constexpr bool value=true;};
template<int N,int...T>constexpr bool all_mappings(std::integer_sequence<int,T...>){return (VerifiedMapping<N,T>::value&&...);}
static_assert(all_mappings<16>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<64>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<32>(std::make_integer_sequence<int,128>{}));
static_assert(all_mappings<128>(std::make_integer_sequence<int,128>{}));
__device__ __forceinline__ void copy16(void* sm,const void* gm){
 unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(sm));
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(gm));
}

// W=128, one head per visit: R/Vr stream in as two 64-row tiles (jt) and the two
// partial softmaxes are merged through `partial`. Grid x starts at query 64; the
// first 64 queries fit a 64-wide window and run split_value_warpgroup<64>.
template<int W,int HV=1,bool STASH=false>
__global__ __launch_bounds__(128) void row_stream_warpgroup(
 const bf16* __restrict__ Q,const bf16* __restrict__ R,const bf16* __restrict__ S,
 const bf16* __restrict__ Vr,const bf16* __restrict__ Vs,bf16* __restrict__ Y,
 float* __restrict__ mout,float* __restrict__ lout,const bool* __restrict__ mask,
 int H,int N,float scale,int win){
 constexpr int G=1,WPH=4,JR=64,VD=64,NT=VD/8,CT=W/8;
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane/4,tig=lane%4;
 const int lh=warp/WPH,th=tid%(WPH*32),i=blockIdx.x+64,group_h0=blockIdx.y*G*HV,b=blockIdx.z;
 const int lo=max(0,i-W+1),hi=min(N,i+1);
 const int row0=warp*16+g,row1=row0+8;
 
 const int64_t kv=(int64_t)b*N*D;
 const bool* mrow=mask+((int64_t)b*N+i)*N;
 using Score=decltype(make_tiled_mma(typename Atom<W>::T{}));
 using Value=decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{}));
 auto al=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<64>{},Int<D>{}));
 auto sl=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<W>{},Int<D>{}));
 auto pl=tile_to_shape(typename PSwizzle<W>::T{},make_shape(Int<64>{},Int<W>{}));
 auto vl=tile_to_shape(GMMA::Layout_MN_SW128_Atom<bf16>{},make_shape(Int<VD>{},Int<W>{}));
 extern __shared__ __align__(128) char storage[];
 bf16* ap=reinterpret_cast<bf16*>(storage);
 bf16* sp=ap+cosize_v<decltype(al)>;
 bf16* pp=ap; // score operand is dead before probability writes
 bf16* vp=sp+cosize_v<decltype(sl)>;
 bf16* rawr=vp+cosize_v<decltype(vl)>;
 bf16* rawvr=rawr+JR*D;
 float* aq=reinterpret_cast<float*>(rawvr+JR*(D+8));
 float* wn=aq+D;
 float* wml=wn+4*VD;
 float* partial=wml+8;
 auto A=make_tensor(make_smem_ptr(ap),al);
 auto B=make_tensor(make_smem_ptr(sp),sl);
 auto P=make_tensor(make_smem_ptr(pp),pl);
 auto V=make_tensor(make_smem_ptr(vp),vl);
 uint4 vstash[STASH ? 2*(W*(VD/8)/128) : 1];
 if constexpr(STASH){
  #pragma unroll
  for(int part=0;part<2;++part){
   #pragma unroll
   for(int k=0;k<W*(VD/8)/128;++k){
    const int idx=tid+k*128,j=idx/(VD/8),d=idx%(VD/8)*8+part*VD,tok=lo+j;
    vstash[part*(W*(VD/8)/128)+k]=tok<N?*reinterpret_cast<const uint4*>(Vs+kv+(int64_t)tok*D+d):make_uint4(0,0,0,0);
   }
  }
 }
 for(int idx=tid;idx<W*(D/8);idx+=128){
  const int j=idx/(D/8),d=idx%(D/8)*8,tok=lo+j;
  if(tok<N){
   const int64_t off=kv+(int64_t)tok*D+d;
   copy16(&B(j,d),S+off);
  }else{
   uint4 z=make_uint4(0,0,0,0);
   *reinterpret_cast<uint4*>(&B(j,d))=z;
  }
 }
 #pragma unroll 1
 for(int jt=0;jt<2;++jt){
  const int j0=lo+jt*JR+row0,j1=lo+jt*JR+row1;
  for(int idx=tid;idx<JR*(D/8);idx+=128){
   const int j=idx/(D/8),d=idx%(D/8)*8,tok=lo+jt*JR+j;
   if(tok<N){
    const int64_t off=kv+(int64_t)tok*D+d;
    copy16(rawr+j*D+d,R+off);copy16(rawvr+j*(D+8)+d,Vr+off);
   }else{
    *reinterpret_cast<uint4*>(rawr+j*D+d)=make_uint4(0,0,0,0);
    *reinterpret_cast<uint4*>(rawvr+j*(D+8)+d)=make_uint4(0,0,0,0);
   }
  }
 #pragma unroll 1
 for(int visit=0;visit<HV;++visit){
 const int h0=group_h0+visit*G;
 for(int idx=tid;idx<G*D;idx+=128){const int head=idx/D,d=idx%D;aq[idx]=scale*float(Q[(((int64_t)b*H+h0+head)*N+i)*D+d]);}
 asm volatile("cp.async.wait_all;"::);
 __syncthreads();
 for(int idx=th;idx<JR*(D/8);idx+=WPH*32){
  const int row=idx/(D/8),d=idx%(D/8)*8;
  uint4 v=*reinterpret_cast<const uint4*>(rawr+row*D+d);
  auto* pairs=reinterpret_cast<__nv_bfloat162*>(&v);
  #pragma unroll
  for(int e=0;e<4;++e){auto f=__bfloat1622float2(pairs[e]);pairs[e]=__floats2bfloat162_rn(aq[lh*D+d+e*2]*f.x,aq[lh*D+d+e*2+1]*f.y);}
  *reinterpret_cast<uint4*>(&A(lh*W+row,d))=v;
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 Score smma;auto st=smma.get_thread_slice(tid);
 auto fa=st.make_fragment_A(st.partition_A(A));auto fb=st.make_fragment_B(st.partition_B(B));
 auto cx=make_identity_tensor(make_shape(Int<64>{},Int<W>{}));auto coord=st.partition_C(cx);auto x=st.make_fragment_C(coord);clear(x);
 warpgroup_fence_operand(x);warpgroup_arrive();gemm(smma,fa,fb,x);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(x);
 float m0=NEG,m1=NEG;
 const bool rm0=j0<hi&&mrow[j0],rm1=j1<hi&&mrow[j1];
 #pragma unroll
 for(int nt=0;nt<CT;++nt){
  const int c0=lo+nt*8+2*tig,c1=c0+1;
  const bool cm0=c0<hi&&mrow[c0],cm1=c1<hi&&mrow[c1];
  x(4*nt)=rm0&&cm0?x(4*nt):NEG;x(4*nt+1)=rm0&&cm1?x(4*nt+1):NEG;
  x(4*nt+2)=rm1&&cm0?x(4*nt+2):NEG;x(4*nt+3)=rm1&&cm1?x(4*nt+3):NEG;
  m0=fmaxf(m0,fmaxf(x(4*nt),x(4*nt+1)));m1=fmaxf(m1,fmaxf(x(4*nt+2),x(4*nt+3)));
 }
 #pragma unroll
 for(int off=1;off<=2;off<<=1){m0=fmaxf(m0,__shfl_xor_sync(0xffffffff,m0,off));m1=fmaxf(m1,__shfl_xor_sync(0xffffffff,m1,off));}
 float l0=0,l1=0;
 #pragma unroll
 for(int nt=0;nt<CT;++nt){
  const int c=nt*8+2*tig;
  const float p0=x(4*nt)<-5e29f?0.f:__expf(x(4*nt)-m0),p1=x(4*nt+1)<-5e29f?0.f:__expf(x(4*nt+1)-m0);
  const float p2=x(4*nt+2)<-5e29f?0.f:__expf(x(4*nt+2)-m1),p3=x(4*nt+3)<-5e29f?0.f:__expf(x(4*nt+3)-m1);
  *reinterpret_cast<__nv_bfloat162*>(&P(row0,c))=__floats2bfloat162_rn(p0,p1);
  *reinterpret_cast<__nv_bfloat162*>(&P(row1,c))=__floats2bfloat162_rn(p2,p3);
  l0+=p0+p1;l1+=p2+p3;
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 #pragma unroll
 for(int off=1;off<=2;off<<=1){l0+=__shfl_xor_sync(0xffffffff,l0,off);l1+=__shfl_xor_sync(0xffffffff,l1,off);}
 float mw=fmaxf(m0,m1);
 #pragma unroll
 for(int off=16;off>0;off>>=1)mw=fmaxf(mw,__shfl_xor_sync(0xffffffff,mw,off));
 const float w0=__expf(m0-mw),w1=__expf(m1-mw);
 float lw=w0*l0+w1*l1;
 #pragma unroll
 for(int off=4;off<=16;off<<=1)lw+=__shfl_xor_sync(0xffffffff,lw,off);
 if(lane==0){wml[warp*2]=mw;wml[warp*2+1]=lw;}

 #pragma unroll
 for(int part=0;part<2;++part){
  #pragma unroll
  for(int k=0;k<W*(VD/8)/128;++k){
   const int idx=tid+k*128,j=idx/(VD/8),d=idx%(VD/8)*8,tok=lo+j;
   if constexpr(STASH){*reinterpret_cast<uint4*>(&V(d,j))=vstash[part*(W*(VD/8)/128)+k];}
   else if(tok<N){copy16(&V(d,j),Vs+kv+(int64_t)tok*D+part*VD+d);}
   else{*reinterpret_cast<uint4*>(&V(d,j))=make_uint4(0,0,0,0);}
  }
  asm volatile("cp.async.wait_all;"::);
  cutlass::arch::fence_view_async_shared();__syncthreads();
  Value vmma;auto vt=vmma.get_thread_slice(tid);
  auto fp=vt.make_fragment_A(vt.partition_A(P));auto fv=vt.make_fragment_B(vt.partition_B(V));
  auto cu=make_identity_tensor(make_shape(Int<64>{},Int<VD>{}));auto uc=vt.partition_C(cu);auto u=vt.make_fragment_C(uc);clear(u);
  warpgroup_fence_operand(u);warpgroup_arrive();gemm(vmma,fp,fv,u);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(u);
 float acc[2*NT];
 #pragma unroll
 for(int nt=0;nt<NT;++nt){
  const int c=part*VD+nt*8+2*tig;
  const auto v0=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawvr+(row0%W)*(D+8)+c));
  const auto v1=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawvr+(row1%W)*(D+8)+c));
  acc[nt*2]=w0*v0.x*u(4*nt)+w1*v1.x*u(4*nt+2);
  acc[nt*2+1]=w0*v0.y*u(4*nt+1)+w1*v1.y*u(4*nt+3);
 }
 #pragma unroll
 for(int off=4;off<=16;off<<=1){
  #pragma unroll
  for(int e=0;e<2*NT;++e)acc[e]+=__shfl_xor_sync(0xffffffff,acc[e],off);
 }
 if(lane<4){
  #pragma unroll
  for(int nt=0;nt<NT;++nt){wn[warp*VD+nt*8+2*lane]=acc[2*nt];wn[warp*VD+nt*8+2*lane+1]=acc[2*nt+1];}
 }
 __syncthreads();
 float mh=NEG;
 #pragma unroll
 for(int w=0;w<WPH;++w)mh=fmaxf(mh,wml[(lh*WPH+w)*2]);
 float l=0;
 #pragma unroll
 for(int w=0;w<WPH;++w)l+=__expf(wml[(lh*WPH+w)*2]-mh)*wml[(lh*WPH+w)*2+1];
 const float oldm=jt?partial[visit*(D+2)+D]:NEG;
 const float oldl=jt?partial[visit*(D+2)+D+1]:0.f;
 const float mergedm=fmaxf(oldm,mh),alpha=__expf(oldm-mergedm),beta=__expf(mh-mergedm);
 const float mergedl=alpha*oldl+beta*l;
 const float inv=mergedl>1e-20f?1.f/mergedl:0.f;const int64_t out=((int64_t)b*H+h0+lh)*N+i;
 for(int d=th;d<VD;d+=WPH*32){float val=0;
  #pragma unroll
  for(int w=0;w<WPH;++w)val+=__expf(wml[(lh*WPH+w)*2]-mh)*wn[(lh*WPH+w)*VD+d];
  const int channel=part*VD+d;
  if(jt==0)partial[visit*(D+2)+channel]=val;
  else Y[out*D+channel]=bf16((alpha*partial[visit*(D+2)+channel]+beta*val)*inv);
 }
 if(th==0 && part==1){
  if(jt==0){partial[visit*(D+2)+D]=mh;partial[visit*(D+2)+D+1]=l;}
  else{mout[out]=mergedm;lout[out]=mergedl;}
 }

  __syncthreads(); // V and wn are reused by the next channel half
 } // channel half
 } // head-group visit
 __syncthreads(); // all heads finished before raw row tile is overwritten
 } // streamed row tile
}

template<int W,int HV>constexpr size_t bytes(){return sizeof(bf16)*(64*D+W*D+(D/2)*W+2*64*D+8*64)+sizeof(float)*(D+4*(D/2)+8+HV*(D+2));}

bool launch(const void* q,const void* r,const void* s,const void* vr,const void* vs,
 void* y,float* m,float* l,const bool* mask,int B,int H,int N,float scale,int max_smem,cudaStream_t stream,int device){
 constexpr size_t smem=bytes<128,8>();
 constexpr size_t edge_smem=att3_shared_hopper_split_impl::bytes<64>();
 if(smem>(size_t)max_smem)return false;
 static thread_local int attribute_device=-1;
 if(attribute_device!=device){
  auto e=cudaFuncSetAttribute(row_stream_warpgroup<128,8,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(e!=cudaSuccess)return false;
  e=cudaFuncSetAttribute(att3_shared_hopper_split_impl::split_value_warpgroup<64,8,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,edge_smem);
  if(e!=cudaSuccess)return false;
  attribute_device=device;
 }
 att3_shared_hopper_split_impl::split_value_warpgroup<64,8,true><<<dim3(64,H/8,B),128,edge_smem,stream>>>(
  reinterpret_cast<const bf16*>(q),reinterpret_cast<const bf16*>(r),reinterpret_cast<const bf16*>(s),
  reinterpret_cast<const bf16*>(vr),reinterpret_cast<const bf16*>(vs),reinterpret_cast<bf16*>(y),m,l,mask,H,N,scale,64);
 row_stream_warpgroup<128,8,true><<<dim3(N-64,H/8,B),128,smem,stream>>>(
  reinterpret_cast<const bf16*>(q),reinterpret_cast<const bf16*>(r),reinterpret_cast<const bf16*>(s),
  reinterpret_cast<const bf16*>(vr),reinterpret_cast<const bf16*>(vs),reinterpret_cast<bf16*>(y),m,l,mask,H,N,scale,128);
 return true;
}

}

bool launch_att3_shared_hopper(const void* q,const void* r,const void* s,const void* vr,const void* vs,
 void* y,float* m,float* l,const bool* mask,int B,int H,int N,float scale,int max_smem,cudaStream_t stream,int win){
 // false: caller falls back to the MMA path, which keeps more parallelism on
 // small grids than one CTA per query and head group.
 if(B<1 || N<128 || N%16 || H<16 || !mask)return false;
 if((win==16 && H%4)||(win==32 && H%2)||((win==64 || win==128) && H%8)||(win!=16 && win!=32 && win!=64 && win!=128))return false;
 int device=0,major=0,minor=0;
 if(cudaGetDevice(&device)!=cudaSuccess || cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess || cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess)return false;
 if(major!=9 || minor!=0)return false;
 using namespace att3_shared_hopper_impl;
 if(win==128){
  if(H<64 || N<256)return false;
  return att3_shared_hopper_w128_impl::launch(q,r,s,vr,vs,y,m,l,mask,B,H,N,scale,max_smem,stream,device);
 }
 if(win==64){
  // split-value only once the grid has >= 2048 CTAs (N*H/8)
  if(H>=64 && (int64_t)H*N>=16384)
   return att3_shared_hopper_split_impl::launch<64,8>(q,r,s,vr,vs,y,m,l,mask,B,H,N,scale,max_smem,stream,device);
  return launch<64,8>(q,r,s,vr,vs,y,m,l,mask,B,H,N,scale,max_smem,stream,device);
 }
 if(win==16)return launch<16,1>(q,r,s,vr,vs,y,m,l,mask,B,H,N,scale,max_smem,stream,device);
 if(H%8==0 && ((H>=64 && N>=128)||(H>=16 && N>=256)))
  return launch<32,4>(q,r,s,vr,vs,y,m,l,mask,B,H,N,scale,max_smem,stream,device);
 return launch<32,1>(q,r,s,vr,vs,y,m,l,mask,B,H,N,scale,max_smem,stream,device);
}
