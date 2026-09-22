#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
// sm_90a WGMMA R/S backward for the shared-KV single gather (D=128, W=64). One CTA
// per anchor a and 4 heads (one warp each); direction 0 writes dR/dVr[a], 1 dS/dVs[a].
// Each 16-query tile scores a K=80-key slice of the staged opposite window.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
namespace att3_shared_rs_wgmma64 {
using namespace cute;
using bf16=cute::bfloat16_t;
constexpr int D=128;
template<int K> struct ScoreAtom;
template<>struct ScoreAtom<32>{using T=SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct ScoreAtom<64>{using T=SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct ScoreAtom<48>{using T=SM90_64x48x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>;};
template<>struct ScoreAtom<80>{using T=SM90_64x80x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>;};
template<int K>struct PAtom;
template<>struct PAtom<32>{using T=GMMA::Layout_K_SW64_Atom<bf16>;};
template<>struct PAtom<64>{using T=GMMA::Layout_K_SW128_Atom<bf16>;};
template<>struct PAtom<48>{using T=GMMA::Layout_K_SW32_Atom<bf16>;};
template<>struct PAtom<80>{using T=GMMA::Layout_K_SW32_Atom<bf16>;};
using Projection=decltype(make_tiled_mma(SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
// check the hand-written accumulator (row, col) indexing against CuTe's layout
template<class Mma,int N,int Tid>constexpr bool mapping_valid(){
 using TV=decltype(Mma{}.get_layoutC_TV());constexpr TV tv{};
 for(int nt=0;nt<N/8;++nt)for(int e=0;e<4;++e){
  const int coord=tv(make_coord(Tid,4*nt+e));
  if(coord%64!=(Tid/32)*16+(Tid%32)/4+(e/2)*8 || coord/64!=nt*8+2*(Tid%4)+(e%2))return false;
 }return true;
}
template<class M,int N,int...T>constexpr bool mappings(std::integer_sequence<int,T...>){return (mapping_valid<M,N,T>()&&...);}
static_assert(mappings<decltype(make_tiled_mma(ScoreAtom<48>::T{})),48>(std::make_integer_sequence<int,128>{}));
static_assert(mappings<decltype(make_tiled_mma(ScoreAtom<80>::T{})),80>(std::make_integer_sequence<int,128>{}));
static_assert(mappings<Projection,128>(std::make_integer_sequence<int,128>{}));
static_assert(mappings<decltype(make_tiled_mma(ScoreAtom<32>::T{})),32>(std::make_integer_sequence<int,128>{}));
static_assert(mappings<decltype(make_tiled_mma(ScoreAtom<64>::T{})),64>(std::make_integer_sequence<int,128>{}));

// score C order equals projection A order, so P/dP feed the projection from registers
template<int K,int Tid>constexpr bool derivative_order(){
 using A=decltype(Projection{}.get_layoutA_TV());constexpr A atv{};
 using M=decltype(make_tiled_mma(typename ScoreAtom<K>::T{}));
 using C=decltype(M{}.get_layoutC_TV());constexpr C ctv{};
 for(int z=0;z<K/2;++z)if(atv(make_coord(Tid,z%8))+64*16*(z/8)!=ctv(make_coord(Tid,z)))return false;
 return true;
}
template<int K,int...T>constexpr bool derivative_orders(std::integer_sequence<int,T...>){return (derivative_order<K,T>()&&...);}
static_assert(derivative_orders<80>(std::make_integer_sequence<int,128>{}));
static_assert(derivative_orders<48>(std::make_integer_sequence<int,128>{}));
static_assert(derivative_orders<32>(std::make_integer_sequence<int,128>{}));
static_assert(derivative_orders<64>(std::make_integer_sequence<int,128>{}));

__device__ __forceinline__ void copy16(void* s,const void* g){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(s));asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(a),"l"(g));}
__device__ __forceinline__ bool masked(const uint32_t* packed,int words,int b,int N,int j,int k){return j<N && k<N && ((packed[((int64_t)b*N+j)*words+k/32]>>(k%32))&1u);}

template<int W,bool SPECIAL,bool PACKED>
__device__ __forceinline__ void work(
 const bf16* R,const bf16* Vr,const bf16* Q,const bf16* dY,const bf16* S,const bf16* Vs,
 const float* m,const float* l,const float* delta,float* dR,float* dVr,float* dS,float* dVs,
 const uint8_t* support,const uint32_t* packed,int H,int N,float scale){
 constexpr int CAP=2*W,K=W+16,NT=D/8,KT=K/8,RPAD=136;
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane/4,tig=lane%4;
 const int a=blockIdx.x,h0=blockIdx.y*4,b=blockIdx.z,lo=max(0,a-W+1),end=min(N,a+W),words=(N+31)/32;
 const int row0=warp*16+g,row1=row0+8;
 using Score=decltype(make_tiled_mma(typename ScoreAtom<K>::T{}));
 auto al=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<64>{},Int<D>{}));
 auto bl=tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},make_shape(Int<CAP>{},Int<D>{}));
 auto pl=tile_to_shape(typename PAtom<K>::T{},make_shape(Int<64>{},Int<K>{}));
 // transposed view of the key/value tile, the projection's B operand
 auto vl=composition(bl,make_layout(make_shape(Int<D>{},Int<CAP>{}),make_stride(Int<CAP>{},Int<1>{})));
 extern __shared__ __align__(128) char storage[];
 bf16* rq=reinterpret_cast<bf16*>(storage),*ry=rq+64*RPAD;
 bf16* kp=ry+64*RPAD,*vp=kp+CAP*D;
 bf16* pp=vp+CAP*D,*gp=pp; // P/GA only shape register fragments; grad overlays this memory
 float* grad=reinterpret_cast<float*>(pp);
 float* anchor=grad+2*4*D;
 auto ql=make_layout(make_shape(Int<64>{},Int<D>{}),make_stride(Int<RPAD>{},Int<1>{}));
 auto RQ=make_tensor(make_smem_ptr(rq),ql),RY=make_tensor(make_smem_ptr(ry),ql);
 auto X=make_tensor(make_smem_ptr(kp),bl),V=make_tensor(make_smem_ptr(vp),bl);
 auto XP=make_tensor(make_smem_ptr(kp),vl),VP=make_tensor(make_smem_ptr(vp),vl);
 auto P=make_tensor(make_smem_ptr(pp),pl),GA=make_tensor(make_smem_ptr(gp),pl);
 Score smma;auto st=smma.get_thread_slice(tid);
 auto tq=st.partition_A(RQ),ty=st.partition_A(RY);
 auto ac=st.partition_A(make_identity_tensor(make_shape(Int<64>{},Int<D>{})));
 auto fa=st.make_fragment_A(tq);
 auto sc=st.partition_C(make_identity_tensor(make_shape(Int<64>{},Int<K>{})));
 Projection pmma;auto pt=pmma.get_thread_slice(tid);
 auto fp=pt.make_fragment_A(pt.partition_A(P)),fg=pt.make_fragment_A(pt.partition_A(GA));
 auto pc=pt.partition_C(make_identity_tensor(make_shape(Int<64>{},Int<D>{})));
 #pragma unroll 1
 for(int direction=0;direction<2;++direction){
  const bf16* ax=direction?S:R,*av=direction?Vs:Vr,*opx=direction?R:S,*opv=direction?Vr:Vs;
  float* ox=direction?dS:dR,*ov=direction?dVs:dVr;
  for(int z=tid;z<2*4*D;z+=128)grad[z]=0;
  for(int d=tid;d<D;d+=128){anchor[d]=scale*float(ax[((int64_t)b*N+a)*D+d]);anchor[D+d]=float(av[((int64_t)b*N+a)*D+d]);}
  for(int z=tid;z<CAP*(D/8);z+=128){int k=z/(D/8),d=z%(D/8)*8;
   if(lo+k<N){copy16(&X(k,d),opx+((int64_t)b*N+lo+k)*D+d);copy16(&V(k,d),opv+((int64_t)b*N+lo+k)*D+d);}
   else{*reinterpret_cast<uint4*>(&X(k,d))=make_uint4(0,0,0,0);*reinterpret_cast<uint4*>(&V(k,d))=make_uint4(0,0,0,0);}
  }
  asm volatile("cp.async.wait_all;"::);__syncthreads();
  for(int j0=a;j0<end;j0+=16){
   const int shift=W>=32?max(0,j0-W+1)-lo:0;
   auto Xj=local_tile(domain_offset(make_coord(shift,0),X),make_tile(Int<K>{},Int<D>{}),make_coord(0,0));
   auto Vj=local_tile(domain_offset(make_coord(shift,0),V),make_tile(Int<K>{},Int<D>{}),make_coord(0,0));
   auto XPj=local_tile(domain_offset(make_coord(0,shift),XP),make_tile(Int<D>{},Int<K>{}),make_coord(0,0));
   auto VPj=local_tile(domain_offset(make_coord(0,shift),VP),make_tile(Int<D>{},Int<K>{}),make_coord(0,0));
   auto fx=st.make_fragment_B(st.partition_B(Xj)),fv=st.make_fragment_B(st.partition_B(Vj));
   auto fpx=pt.make_fragment_B(pt.partition_B(XPj)),fpv=pt.make_fragment_B(pt.partition_B(VPj));

   for(int z=tid;z<64*(D/8);z+=128){int row=z/(D/8),d=z%(D/8)*8,j=j0+row%16,head=h0+row/16;
    if(j<N && j<end){int64_t o=(((int64_t)b*H+head)*N+j)*D+d;copy16(rq+row*RPAD+d,Q+o);copy16(ry+row*RPAD+d,dY+o);}
    else{*reinterpret_cast<uint4*>(rq+row*RPAD+d)=make_uint4(0,0,0,0);*reinterpret_cast<uint4*>(ry+row*RPAD+d)=make_uint4(0,0,0,0);}
   }
   asm volatile("cp.async.wait_all;"::);__syncthreads();
   #pragma unroll
   for(int z=0;z<size(fa);++z)fa(z)=bf16(float(tq(z))*anchor[get<1>(ac(z))]);
   warpgroup_fence_operand(fa);
   auto score=st.make_fragment_C(sc);clear(score);
   warpgroup_fence_operand(score);warpgroup_arrive();gemm(smma,fa,fx,score);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(score);
   #pragma unroll
   for(int z=0;z<size(fa);++z)fa(z)=bf16(float(ty(z))*anchor[D+get<1>(ac(z))]);
   warpgroup_fence_operand(fa);
   auto adr=st.make_fragment_C(sc);clear(adr);
   warpgroup_fence_operand(adr);warpgroup_arrive();gemm(smma,fa,fv,adr);warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(adr);
   const int jx[2]={j0+g,j0+g+8};float mr[2],il[2],ds[2];bool single[2];
   #pragma unroll
   for(int r=0;r<2;++r){int j=jx[r];bool valid=j<end && masked(packed,words,b,N,j,a);int64_t o=((int64_t)b*H+h0+warp)*N+min(j,N-1);mr[r]=m[o];ds[r]=delta[o];il[r]=valid?1.f/fmaxf(l[o],1.e-20f):0.f;single[r]=SPECIAL && j<N && support[(int64_t)b*N+j]==1;}
   #pragma unroll
   for(int nt=0;nt<KT;++nt){int k=nt*8+2*tig;
    #pragma unroll
    for(int r=0;r<2;++r){float pr[2],ga[2];
     #pragma unroll
     for(int e=0;e<2;++e){int f=4*nt+2*r+e;bool on=masked(packed,words,b,N,jx[r],lo+shift+k+e);float inv=on?il[r]:0.f;pr[e]=single[r]?(inv>0.f?1.f:0.f):__expf(fminf(score(f)-mr[r],0.f))*inv;ga[e]=single[r]?0.f:(adr(f)-ds[r])*pr[e];}
     fp(4*nt+2*r)=bf16(pr[0]);fp(4*nt+2*r+1)=bf16(pr[1]);fg(4*nt+2*r)=bf16(ga[0]);fg(4*nt+2*r+1)=bf16(ga[1]);
    }
   }
   warpgroup_fence_operand(fp);warpgroup_fence_operand(fg);
   // one 64-float projection accumulator live at a time, for register pressure
   #pragma unroll 1
   for(int kind=0;kind<2;++kind){
    auto u=pt.make_fragment_C(pc);clear(u);
    warpgroup_fence_operand(u);warpgroup_arrive();
    if(kind==0)gemm(pmma,fg,fpx,u);else gemm(pmma,fp,fpv,u);
    warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(u);
    // P/gA live in registers and rq/ry are no longer read, so their storage
    // holds the FP32 u tile; the barrier orders all warps' last rq/ry reads
    // before the first overwrite. Rows are XOR-swizzled in 8-float chunks.
    if(kind==0)__syncthreads();
    float* scratch=reinterpret_cast<float*>(storage);
    #pragma unroll
    for(int nt=0;nt<NT;++nt){int d=nt*8+2*tig;
     *reinterpret_cast<float2*>(scratch+row0*D+(d^((row0&3)*8)))=make_float2(u(4*nt),u(4*nt+1));
     *reinterpret_cast<float2*>(scratch+row1*D+(d^((row1&3)*8)))=make_float2(u(4*nt+2),u(4*nt+3));
    }
    // each warp only touches its own head's 16 rows, so a warp barrier suffices
    __syncwarp();
    const int d=lane*4;
    float4 accum=make_float4(0.f,0.f,0.f,0.f);
    const bf16* input=kind?dY:Q;
    const int64_t head_offset=((int64_t)b*H+h0+warp)*N*D;
    #pragma unroll
    for(int jr=0;jr<16;++jr){int j=j0+jr,row=warp*16+jr;
     const float4 value=*reinterpret_cast<const float4*>(scratch+row*D+(d^((row&3)*8)));
     float2 x0=make_float2(0.f,0.f),x1=make_float2(0.f,0.f);
     if(j<end){const auto* p=input+head_offset+(int64_t)j*D+d;x0=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));x1=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p+2));}
     accum.x=fmaf(x0.x,value.x,accum.x);accum.y=fmaf(x0.y,value.y,accum.y);
     accum.z=fmaf(x1.x,value.z,accum.z);accum.w=fmaf(x1.y,value.w,accum.w);
    }
    grad[(kind*4+warp)*D+d]+=accum.x;grad[(kind*4+warp)*D+d+1]+=accum.y;
    grad[(kind*4+warp)*D+d+2]+=accum.z;grad[(kind*4+warp)*D+d+3]+=accum.w;
    __syncwarp();
   }
   __syncthreads();
  }
  for(int z=tid;z<4*D;z+=128){int head=z/D,d=z%D;int64_t o=(((int64_t)b*H+h0+head)*N+a)*D+d;if constexpr(PACKED){reinterpret_cast<__nv_bfloat16*>(ox)[o]=__float2bfloat16_rn(scale*grad[z]);reinterpret_cast<__nv_bfloat16*>(ov)[o]=__float2bfloat16_rn(grad[4*D+z]);}else{ox[o]=scale*grad[z];ov[o]=grad[4*D+z];}}
  __syncthreads();
 }
}
template<int W,bool PACKED=false>__global__ __launch_bounds__(128) void kernel(
 const bf16* R,const bf16* Vr,const bf16* Q,const bf16* dY,const bf16* S,const bf16* Vs,
 const float* m,const float* l,const float* delta,float* dR,float* dVr,float* dS,float* dVs,
 const uint8_t* support,const uint32_t* packed,int H,int N,float scale){
 int j=blockIdx.x+threadIdx.x;bool special=__syncthreads_or(j<min(N,(int)blockIdx.x+W) && support[(int64_t)blockIdx.z*N+j]==1);
 if(special)work<W,true,PACKED>(R,Vr,Q,dY,S,Vs,m,l,delta,dR,dVr,dS,dVs,support,packed,H,N,scale);
 else work<W,false,PACKED>(R,Vr,Q,dY,S,Vs,m,l,delta,dR,dVr,dS,dVs,support,packed,H,N,scale);
}
template<int W>constexpr int bytes(){return 2*(2*64*136+2*(2*W)*D)+4*(2*4*D+2*D);}
template<bool PACKED>int configure(){
 static thread_local int last_device=-1;int device=-1;auto e=cudaGetDevice(&device);if(e)return int(e);
 if(last_device!=device){e=cudaFuncSetAttribute(kernel<64,PACKED>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes<64>());if(e)return int(e);last_device=device;}
 return 0;
}
} // namespace att3_shared_rs_wgmma64
extern "C" int att3_shared_rs_wgmma64_info(int* info,bool packed_partials){
 int e=packed_partials?att3_shared_rs_wgmma64::configure<true>():att3_shared_rs_wgmma64::configure<false>();if(e)return e;
 const void* f=packed_partials?(const void*)att3_shared_rs_wgmma64::kernel<64,true>:(const void*)att3_shared_rs_wgmma64::kernel<64,false>;
 cudaFuncAttributes a;e=cudaFuncGetAttributes(&a,f);if(e)return e;int active=0;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,128,att3_shared_rs_wgmma64::bytes<64>());
 info[0]=a.numRegs;info[1]=a.localSizeBytes;info[2]=att3_shared_rs_wgmma64::bytes<64>();info[3]=active;info[4]=128;return e;
}
extern "C" int att3_shared_rs_wgmma64_w64(
 const __nv_bfloat16* R,const __nv_bfloat16* Vr,const __nv_bfloat16* Q,const __nv_bfloat16* dY,const __nv_bfloat16* S,const __nv_bfloat16* Vs,
 const float* m,const float* l,const float* delta,void* dR,void* dVr,void* dS,void* dVs,
 int B,int H,int N,int win,float scale,cudaStream_t stream,const uint8_t* support,const uint32_t* packed,bool packed_partials){
 if(win!=64 || H%4 || B<1 || N<1 || !support || !packed)return cudaErrorInvalidValue;
 int e=packed_partials?att3_shared_rs_wgmma64::configure<true>():att3_shared_rs_wgmma64::configure<false>();if(e)return e;
 const void* f=packed_partials?(const void*)att3_shared_rs_wgmma64::kernel<64,true>:(const void*)att3_shared_rs_wgmma64::kernel<64,false>;
 void* args[]={&R,&Vr,&Q,&dY,&S,&Vs,&m,&l,&delta,&dR,&dVr,&dS,&dVs,&support,&packed,&H,&N,&scale};
 return cudaLaunchKernel(f,dim3(N,H/4,B),dim3(128),args,att3_shared_rs_wgmma64::bytes<64>(),stream);
}
