#include <stdexcept>
// sm_90a WGMMA dQ for the shared-KV single gather (D=128, W in {16,32,64}). One CTA
// per query a; G=64/W heads fill the 64 MMA rows, reusing one staged R/Vr/S/Vs window.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
namespace dq_wgmma {
using namespace cute;
using bf16=cute::bfloat16_t;
constexpr int D=128,RP=136;
template<int W,bool REG>struct ScoreAtom;
#define ATOM(W) template<>struct ScoreAtom<W,false>{using T=SM90_64x##W##x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>;}; template<>struct ScoreAtom<W,true>{using T=SM90_64x##W##x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>;};
ATOM(16) ATOM(32) ATOM(64)
#undef ATOM
using Projection=decltype(make_tiled_mma(SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
// check the hand-written accumulator (row, col) indexing against CuTe's layout
template<class Mma,int N,int Tid>constexpr bool mapping_valid(){
 using TV=decltype(Mma{}.get_layoutC_TV());constexpr TV tv{};
 for(int nt=0;nt<N/8;++nt)for(int e=0;e<4;++e){
  int c=tv(make_coord(Tid,4*nt+e));
  if(c%64!=(Tid/32)*16+(Tid%32)/4+(e/2)*8 || c/64!=nt*8+2*(Tid%4)+e%2)return false;
 }return true;
}
template<class M,int N,int...T>constexpr bool mappings(std::integer_sequence<int,T...>){return (mapping_valid<M,N,T>()&&...);}
static_assert(mappings<Projection,128>(std::make_integer_sequence<int,128>{}));
static_assert(mappings<decltype(make_tiled_mma(ScoreAtom<32,false>::T{})),32>(std::make_integer_sequence<int,128>{}));
// the score accumulator's register order equals the projection's A-fragment order,
// so dP can be fed to the projection MMA from registers without a shuffle
template<int W,int Tid>constexpr bool derivative_order(){
 using A=decltype(Projection{}.get_layoutA_TV());constexpr A atv{};
 using S=decltype(make_tiled_mma(typename ScoreAtom<W,true>::T{}));
 using C=decltype(S{}.get_layoutC_TV());constexpr C ctv{};
 for(int z=0;z<W/2;++z)if(atv(make_coord(Tid,z%8))+64*16*(z/8)!=ctv(make_coord(Tid,z)))return false;
 return true;
}
template<int W,int...T>constexpr bool derivative_orders(std::integer_sequence<int,T...>){return (derivative_order<W,T>()&&...);}
static_assert(derivative_orders<16>(std::make_integer_sequence<int,128>{}));
static_assert(derivative_orders<32>(std::make_integer_sequence<int,128>{}));
static_assert(derivative_orders<64>(std::make_integer_sequence<int,128>{}));
__device__ __forceinline__ void copy16(void* s,const void* g){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(s));asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(a),"l"(g));}

template<int W,int HV,bool REG_SCORE,bool OVERLAP=false>
__global__ __launch_bounds__(128) void kernel(
 const bf16* __restrict__ Q,const bf16* __restrict__ dY,const bf16* __restrict__ R,
 const bf16* __restrict__ Vr,const bf16* __restrict__ S,const bf16* __restrict__ Vs,
 const float* __restrict__ m,const float* __restrict__ l,const float* __restrict__ delta,
 float* __restrict__ dq,const bool* __restrict__ mask,const uint8_t* support,int H,int N,float scale){
 constexpr int G=64/W,WPH=W/16,NT=D/8;
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane/4,tig=lane%4;
 const int lh=warp/WPH,th=tid%(WPH*32),a=blockIdx.x,hbase=blockIdx.y*G*HV,b=blockIdx.z;
 const int lo=max(0,a-W+1),hi=min(N,a+1),row0=warp*16+g,row1=row0+8;
 const int j0=lo+row0%W,j1=lo+row1%W;
 const bool* mrow=mask+((int64_t)b*N+a)*N;
 // a query with at most one visible key has a constant softmax, so dQ is zero
 int count;
 if(support)count=support[(int64_t)b*N+a];
 else{count=0;for(int k=lo;k<hi;k+=32)count+=__popc(__ballot_sync(0xffffffff,k+lane<hi&&mrow[k+lane]));}
 if(count<=1){
  for(int z=tid;z<G*HV*D;z+=128)dq[(((int64_t)b*H+hbase+z/D)*N+a)*D+z%D]=0.f;
  return;
 }
 using Score=decltype(make_tiled_mma(typename ScoreAtom<W,REG_SCORE>::T{}));
 auto al=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<64>{},Int<D>{}));
 auto bl=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<W>{},Int<D>{}));
 auto vl=composition(bl,make_layout(make_shape(Int<D>{},Int<W>{}),make_stride(Int<W>{},Int<1>{})));
 auto gl=make_layout(make_shape(Int<64>{},Int<W>{}),make_stride(Int<W>{},Int<1>{}));
 extern __shared__ __align__(128) char storage[];
 bf16* rawr=reinterpret_cast<bf16*>(storage),*rawvr=rawr+W*RP;
 bf16* sp=rawvr+W*RP,*vp=sp+W*D;
 bf16* ap=vp+W*D,*yp=ap+(REG_SCORE?0:64*D);
 float* anchor=reinterpret_cast<float*>(yp+(REG_SCORE?0:64*D));
 float* wn=anchor+2*G*D;
 auto X=make_tensor(make_smem_ptr(sp),bl),V=make_tensor(make_smem_ptr(vp),bl);
 auto XP=make_tensor(make_smem_ptr(sp),vl);
 auto A=make_tensor(make_smem_ptr(ap),al),Y=make_tensor(make_smem_ptr(yp),al);
 auto GA=make_tensor(make_smem_ptr(ap),gl); // shape carrier for register fragments; never dereferenced
 Score smma;auto st=smma.get_thread_slice(tid);
 auto fx=st.make_fragment_B(st.partition_B(X)),fv=st.make_fragment_B(st.partition_B(V));
 auto ac=st.partition_A(make_identity_tensor(make_shape(Int<64>{},Int<D>{})));
 auto sc=st.partition_C(make_identity_tensor(make_shape(Int<64>{},Int<W>{})));
 Projection pmma;auto pt=pmma.get_thread_slice(tid);
 auto fg=pt.make_fragment_A(pt.partition_A(GA));
 auto gc=pt.partition_A(make_identity_tensor(make_shape(Int<64>{},Int<W>{})));
 auto fpx=pt.make_fragment_B(pt.partition_B(XP));
 auto pc=pt.partition_C(make_identity_tensor(make_shape(Int<64>{},Int<D>{})));
 for(int z=tid;z<W*(D/8);z+=128){int row=z/(D/8),d=z%(D/8)*8,tok=lo+row;
  if(tok<N){const int64_t off=((int64_t)b*N+tok)*D+d;copy16(rawr+row*RP+d,R+off);copy16(rawvr+row*RP+d,Vr+off);copy16(&X(row,d),S+off);copy16(&V(row,d),Vs+off);}
  else{uint4 zero=make_uint4(0,0,0,0);*reinterpret_cast<uint4*>(rawr+row*RP+d)=zero;*reinterpret_cast<uint4*>(rawvr+row*RP+d)=zero;*reinterpret_cast<uint4*>(&X(row,d))=zero;*reinterpret_cast<uint4*>(&V(row,d))=zero;}
 }
 asm volatile("cp.async.wait_all;"::);__syncthreads();
 #pragma unroll 1
 for(int visit=0;visit<HV;++visit){
  int h0=hbase+visit*G;
  for(int z=tid;z<G*D;z+=128){int head=z/D,d=z%D;int64_t off=(((int64_t)b*H+h0+head)*N+a)*D+d;anchor[z]=scale*float(Q[off]);anchor[G*D+z]=float(dY[off]);}
  __syncthreads();
  auto score=st.make_fragment_C(sc),gp=st.make_fragment_C(sc);clear(score);clear(gp);
  if constexpr(REG_SCORE){
   auto fa=st.make_fragment_A(st.partition_A(A));
   #pragma unroll
   for(int z=0;z<size(fa);++z){int row=get<0>(ac(z)),d=get<1>(ac(z));fa(z)=bf16(float(rawr[(row%W)*RP+d])*anchor[(row/W)*D+d]);}
   warpgroup_fence_operand(fa);warpgroup_fence_operand(score);warpgroup_arrive();gemm(smma,fa,fx,score);warpgroup_commit_batch();
   if constexpr(OVERLAP){
    // separate fragment keeps score's A alive; build dY*Vr while the score MMA runs
    auto fy=st.make_fragment_A(st.partition_A(A));
    #pragma unroll
    for(int z=0;z<size(fy);++z){int row=get<0>(ac(z)),d=get<1>(ac(z));fy(z)=bf16(float(rawvr[(row%W)*RP+d])*anchor[G*D+(row/W)*D+d]);}
    warpgroup_fence_operand(fy);warpgroup_fence_operand(gp);warpgroup_arrive();gemm(smma,fy,fv,gp);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(gp);warpgroup_fence_operand(score);
   }else{
    warpgroup_wait<0>();warpgroup_fence_operand(score);
    #pragma unroll
    for(int z=0;z<size(fa);++z){int row=get<0>(ac(z)),d=get<1>(ac(z));fa(z)=bf16(float(rawvr[(row%W)*RP+d])*anchor[G*D+(row/W)*D+d]);}
    warpgroup_fence_operand(fa);warpgroup_fence_operand(gp);warpgroup_arrive();gemm(smma,fa,fv,gp);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(gp);
   }
  }else{
   for(int z=th;z<W*(D/8);z+=WPH*32){int row=z/(D/8),d=z%(D/8)*8;
    uint4 rv=*reinterpret_cast<uint4*>(rawr+row*RP+d),vv=*reinterpret_cast<uint4*>(rawvr+row*RP+d);
    auto rp=reinterpret_cast<__nv_bfloat162*>(&rv),vp2=reinterpret_cast<__nv_bfloat162*>(&vv);
    #pragma unroll
    for(int e=0;e<4;++e){auto rf=__bfloat1622float2(rp[e]),vf=__bfloat1622float2(vp2[e]);rp[e]=__floats2bfloat162_rn(anchor[lh*D+d+e*2]*rf.x,anchor[lh*D+d+e*2+1]*rf.y);vp2[e]=__floats2bfloat162_rn(anchor[G*D+lh*D+d+e*2]*vf.x,anchor[G*D+lh*D+d+e*2+1]*vf.y);}
    *reinterpret_cast<uint4*>(&A(lh*W+row,d))=rv;*reinterpret_cast<uint4*>(&Y(lh*W+row,d))=vv;
   }
   cutlass::arch::fence_view_async_shared();__syncthreads();
   auto fa=st.make_fragment_A(st.partition_A(A)),fy=st.make_fragment_A(st.partition_A(Y));
   warpgroup_fence_operand(score);warpgroup_fence_operand(gp);warpgroup_arrive();gemm(smma,fa,fx,score);gemm(smma,fy,fv,gp);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(score);warpgroup_fence_operand(gp);
  }
  int64_t off=((int64_t)b*H+h0+lh)*N+a;
  float ma=m[off],ila=1.f/fmaxf(l[off],1.e-20f),ds=delta[off];
  #pragma unroll
  for(int z=0;z<size(fg);++z){
   int row=get<0>(gc(z)),col=get<1>(gc(z));
   // derivative_order: C score order equals projection A order, including K slices
   constexpr bool DIRECT=true;
   int f=DIRECT?z:(col/8)*4+((row%16)/8)*2+col%2;
   int j=lo+row%W,k=lo+col;
   float il=(j<hi && k<hi && mrow[j] && mrow[k])?ila:0.f;
   fg(z)=bf16((gp(f)-ds)*(__expf(fminf(score(f)-ma,0.f))*il));
  }
  auto u=pt.make_fragment_C(pc);clear(u);
  warpgroup_fence_operand(fg);warpgroup_fence_operand(u);warpgroup_arrive();gemm(pmma,fg,fpx,u);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(u);
  #pragma unroll
  for(int nt=0;nt<NT;++nt){int d=nt*8+2*tig;
   auto r0=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawr+(row0%W)*RP+d));auto r1=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(rawr+(row1%W)*RP+d));
   float v0=r0.x*u(4*nt)+r1.x*u(4*nt+2),v1=r0.y*u(4*nt+1)+r1.y*u(4*nt+3);
   #pragma unroll
   for(int dlt=4;dlt<=16;dlt*=2){v0+=__shfl_xor_sync(0xffffffff,v0,dlt);v1+=__shfl_xor_sync(0xffffffff,v1,dlt);}
   if(lane<4){wn[warp*D+d]=v0;wn[warp*D+d+1]=v1;}
  }
  __syncthreads();
  for(int d=th;d<D;d+=WPH*32){float val=0.f;
   if constexpr(WPH==4){
    // fixed pairwise tree over the four 16-row warps
    val=(wn[d]+wn[D+d])+(wn[2*D+d]+wn[3*D+d]);
   }else{
    #pragma unroll
    for(int w=0;w<WPH;++w)val+=wn[(lh*WPH+w)*D+d];
   }
   dq[off*D+d]=scale*val;
  }
  __syncthreads();
 }
}
template<int W,int HV,bool R>constexpr int bytes(){return sizeof(bf16)*(2*W*RP+2*W*D+(R?0:2*64*D))+sizeof(float)*(2*(64/W)*D+4*D);}
} // namespace dq_wgmma

namespace dq_wgmma {
template<int W>bool launch(const void* q,const void* dy,const void* r,const void* vr,const void* s,const void* vs,
 const float* m,const float* l,const float* delta,float* dq,const bool* mask,int B,int H,int N,
 float scale,cudaStream_t stream,const uint8_t* support,int device){
 constexpr int smem=bytes<W,1,true>();
 auto fn=kernel<W,1,true,false>;
 static thread_local int attribute_device=-1;
 if(attribute_device!=device){auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));attribute_device=device;}
 fn<<<dim3(N,H/(64/W),B),128,smem,stream>>>(
  static_cast<const bf16*>(q),static_cast<const bf16*>(dy),static_cast<const bf16*>(r),static_cast<const bf16*>(vr),
  static_cast<const bf16*>(s),static_cast<const bf16*>(vs),m,l,delta,dq,mask,support,H,N,scale);
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));return true;
}
}
// false: shape unsupported here, caller falls back to the MMA path
bool launch_att3_shared_hopper_dq(const void* q,const void* dy,const void* r,const void* vr,const void* s,const void* vs,
 const float* m,const float* l,const float* delta,float* dq,const bool* mask,int B,int H,int N,int win,
 float scale,cudaStream_t stream,const uint8_t* support){
 if(B<1 || H<16 || H%4 || N<128 || N%16 || !mask || (win!=16 && win!=32 && win!=64))return false;
 int device=0;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
 static thread_local int checked_device=-1;static thread_local bool compatible=false;
 if(checked_device!=device){int major=0,minor=0;cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);compatible=major==9 && minor==0;checked_device=device;}
 if(!compatible)return false;
 using namespace dq_wgmma;
 if(win==16)return launch<16>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,scale,stream,support,device);
 if(win==32)return launch<32>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,scale,stream,support,device);
 return launch<64>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,scale,stream,support,device);
}
