#pragma once
#include "../common.cuh"
#include <stdexcept>
namespace sg_retained_dq {

// single_col drops the second column buffer: the caller promises win <= bk,
// so there is exactly one column tile.
__host__ __device__ constexpr size_t btc_smem_bytes(int D, int warps, int bk, bool single_col) {
    const int bj = warps * 16, dpad = D + 8;
    const int cb = single_col ? 1 : 2;
    return sizeof(bf16) * ((size_t)2 * bj * dpad + (size_t)cb * 2 * bk * dpad)
         + sizeof(float) * ((size_t)3 * D + cb * 3 * bk + 3 * bj + warps * D + D)
         + sizeof(float) * (cb * (size_t)bk + bj);
}

// backward.cu's Bwd_gather_tc query-anchor pass specialized for the shared-KV
// dQ (D=128, Hkv=1, masked): HG heads per CTA, run sequentially. CACHE selects
// the raw operands kept in shared memory: 0 none (epilogue re-reads rows from
// global), 1 the current row tile, 5 R/Vr/S/Vs with S/Vs read in place. With
// CACHE 5 and HG > 1 the next head's Q/dY anchor row is prefetched during the
// current head. Queries with at most one visible key get exact zero dQ.
template<int WARPS, int BK, bool SINGLE_COL = false, int CACHE = 0, int HG = 1>
__global__ __launch_bounds__(WARPS * 32, 1)
void Bwd_gather_tc(
    const bf16* __restrict__ Xa_bf,  // anchor side [B,H,N,D]
    const bf16* __restrict__ gYa_bf,
    const bf16* __restrict__ Xr_bf,  // row side
    const bf16* __restrict__ Vr_bf,
    const bf16* __restrict__ Xc_bf,  // col side (streamed per k tile)
    const bf16* __restrict__ Vc_bf,
    const float* __restrict__ m_a, const float* __restrict__ l_a, const float* __restrict__ sum_a,
    float* __restrict__ gradXa,      // [B,H,N,D] fp32, direct store
    const bool* __restrict__ mask,   // [B,N,N]
    int H, int N, int win, float scale, int Hkv, const uint8_t* support_meta)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128;
    constexpr int DPAD = D + 8;
    constexpr int BJ = WARPS * 16;
    constexpr int KS = D / 16;      // score GEMM k-steps (D contracted)
    constexpr bool PREFETCH = CACHE == 5 && HG > 1;

    const int a = blockIdx.x;
    int query_support=0;
    for (int qi=0; qi<HG && a<N; ++qi) {
    const int h = blockIdx.y*HG+qi;
    const int b = blockIdx.z;

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g    = lane / 4;
    const int tig  = lane % 4;
    const int lrow  = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int lcol8 = (lane >> 4) * 8;
    const int brow  = (lane & 7) + ((lane >> 4) & 1) * 8;
    const int bcol8 = ((lane >> 3) & 1) * 8;

    constexpr int CB = SINGLE_COL ? 1 : 2;  // column buffer count

    const int COLCAP = CACHE==5 ? ((((win+max(BJ,BK)-1)/max(BJ,BK))*max(BJ,BK)+15)/16)*16 : CB*BK;
    extern __shared__ char smem_raw[];
    // The two A operands, anchor already folded in: scale*Xa o Xr, gYa o Vr.
    // Staged per row block.
    bf16* a0_sm   = reinterpret_cast<bf16*>(smem_raw);            // [BJ][DPAD]
    bf16* a1_sm   = a0_sm + BJ * DPAD;
    bf16* xc_sm   = a1_sm + BJ * DPAD;                            // [CB][BK][DPAD]
    bf16* vc_sm   = xc_sm + COLCAP * DPAD;
    float* anchX  = reinterpret_cast<float*>(vc_sm + COLCAP * DPAD);  // [D] scale*Xa
    float* anchG  = anchX + 2 * D;                                // [D]
    float* wOut   = anchG + D + 3 * CB * BK + 3 * BJ;             // [WARPS][D]
    float* redOut = wOut + WARPS * D;                            // [D]
    // The tile in flight's mask window.
    float* ilac_sm = redOut + D;                                         // [CB][BK]
    float* par_sm  = ilac_sm + CB * BK;                                  // [BJ]

    // Q, dY, the forward stats and dQ live at the query head (b*H + h); the
    // key/value rows and cols at the KV head (b*Hkv + h / (H/Hkv)).
    const int64_t bh = (int64_t)b * H + h;
    const int64_t kvh = (int64_t)b * Hkv + h / (H / Hkv);
    const int64_t q_off = bh * N * D, kv_off = kvh * N * D;
    const int64_t anc_off  = q_off + (int64_t)a * D;              // Xa / gYa reads, gradXa stores
    const int64_t rows_off = kv_off;                              // Xr / Vr
    const int64_t nd_off   = kv_off;                              // Xc / Vc
    const int64_t st_off = bh * N;

    if(qi==0) query_support=support_meta[(int64_t)b*N+a];
    if(query_support<=1) {
        for(int d=tid;d<D;d+=blockDim.x) gradXa[anc_off+d]=0.0f;
        continue;
    }
    constexpr int DV = D / 8;
    const int cache_lo = max(0, a - win + 1);
    const int cache_len = ((((win + max(BJ,BK)-1)/max(BJ,BK))*max(BJ,BK)+15)/16)*16;
    bf16* rawR = reinterpret_cast<bf16*>(smem_raw + btc_smem_bytes(D,WARPS,BK,SINGLE_COL) + 2*(COLCAP-CB*BK)*DPAD*sizeof(bf16));
    bf16* rawV = rawR + cache_len * DPAD;
    if constexpr (CACHE == 5) {
        if(qi==0) {
            for(int idx=tid;idx<cache_len*DV;idx+=blockDim.x) {
                const int r=idx/DV,d=(idx%DV)*8,j=cache_lo+r;
                if(j<N) {
                    cp_async16(rawR+r*DPAD+d,Xr_bf+rows_off+(int64_t)j*D+d);
                    cp_async16(rawV+r*DPAD+d,Vr_bf+rows_off+(int64_t)j*D+d);
                    cp_async16(xc_sm+r*DPAD+d,Xc_bf+nd_off+(int64_t)j*D+d);
                    cp_async16(vc_sm+r*DPAD+d,Vc_bf+nd_off+(int64_t)j*D+d);
                } else {
                    uint4 z=make_uint4(0,0,0,0);
                    *reinterpret_cast<uint4*>(rawR+r*DPAD+d)=z;
                    *reinterpret_cast<uint4*>(rawV+r*DPAD+d)=z;
                    *reinterpret_cast<uint4*>(xc_sm+r*DPAD+d)=z;
                    *reinterpret_cast<uint4*>(vc_sm+r*DPAD+d)=z;
                }
            }
            asm volatile("cp.async.wait_all;" ::);
            __syncthreads();
        }
    }

    bf16* nextQ = rawV+cache_len*DPAD;
    bf16* nextDY = nextQ+D;
    // ---- one-time loads: anchor only (the col side stages per k tile) ----
    for (int d = tid; d < D; d += blockDim.x) {
        anchX[d] = scale * bf2f(PREFETCH && qi>0 ? nextQ[d] : Xa_bf[anc_off+d]);
        anchG[d] = bf2f(PREFETCH && qi>0 ? nextDY[d] : gYa_bf[anc_off+d]);
    }
    for (int d = tid; d < D; d += blockDim.x) redOut[d] = 0.0f;

    const float ma  = m_a[st_off + a];
    const float ila = 1.0f / fmaxf(l_a[st_off + a], DENOM_EPS);
    const float sa  = sum_a[st_off + a];
    const bool* mb  = mask + (int64_t)b * N * N;

    // ---- row blocks of BJ rows, one 16-row tile per warp ----
    // win > 0 restricts the row blocks and, per block, the col tiles to those
    // that can hold visible pairs (sg_row_bounds / sg_col_bounds); the mask
    // still decides every cell.
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ANCHOR, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();  // previous iteration's smem reads (and anchor) done
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ANCHOR, a, j0, BJ, N, win, BK, k_lo, k_hi);

        // Stage col tile k0 into buffer `buf`: matrices and the anchor's mask
        // row folded into its inv-l. Zero-filled pads carry a zero inv-l,
        // gating the tail tile off with no per-cell test.
        auto stage_cols = [&](int k0, int buf) {
            bf16* xs = xc_sm + buf * BK * DPAD;
            bf16* vs = vc_sm + buf * BK * DPAD;
            if constexpr(CACHE != 5) { for (int idx = tid; idx < BK * DV; idx += blockDim.x) {
                const int kl = idx / DV, dv = (idx % DV) * 8;
                const int k = k0 + kl;
                if (k < N) {
                    const int64_t off = nd_off + (int64_t)k * D + dv;
                    cp_async16(xs + kl * DPAD + dv, Xc_bf + off);
                    cp_async16(vs + kl * DPAD + dv, Vc_bf + off);
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(xs + kl * DPAD + dv) = z;
                    *reinterpret_cast<uint4*>(vs + kl * DPAD + dv) = z;
                }
            }
            }
            for (int kl = tid; kl < BK; kl += blockDim.x) {
                const int k = k0 + kl;
                ilac_sm[buf * BK + kl] =
                    (k < N && mb[(int64_t)a * N + k]) ? ila : 0.0f;
            }
        };

        // A operands: the anchor rescale is done in fp32 with a single bf16
        // rounding, matching Y_gather_tc's Qp precision. Pad rows stage as
        // zeros.
        for (int idx = tid; idx < BJ * DV; idx += blockDim.x) {
            const int jl = idx / DV, dv = (idx % DV) * 8;
            const int j = j0 + jl;
            uint4 xq = make_uint4(0, 0, 0, 0), vq = xq;
            if (j < N) {
                const int64_t off = rows_off + (int64_t)j * D + dv;
                xq = *reinterpret_cast<const uint4*>(CACHE==5 ? rawR+(j-cache_lo)*DPAD+dv : Xr_bf+off);
                vq = *reinterpret_cast<const uint4*>(CACHE==5 ? rawV+(j-cache_lo)*DPAD+dv : Vr_bf+off);
            }
            if constexpr(CACHE==1) *reinterpret_cast<uint4*>(rawR+jl*DPAD+dv)=xq;
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xq);
            const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vq);
            uint4 a0, a1;
            __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&a0);
            __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(&a1);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int d = dv + 2 * e;
                const float2 x = __bfloat1622float2(xp[e]);
                const float2 v = __bfloat1622float2(vp[e]);
                p0[e] = __floats2bfloat162_rn(x.x * anchX[d], x.y * anchX[d + 1]);
                p1[e] = __floats2bfloat162_rn(v.x * anchG[d], v.y * anchG[d + 1]);
            }
            *reinterpret_cast<uint4*>(a0_sm + jl * DPAD + dv) = a0;
            *reinterpret_cast<uint4*>(a1_sm + jl * DPAD + dv) = a1;
        }
        for (int jl = tid; jl < BJ; jl += blockDim.x) {
            const int j = j0 + jl;
            // Staged rather than kept in registers: a live register costs more
            // than a rematerializable smem read.
            if (j < N) par_sm[jl] = mb[(int64_t)a * N + j] ? 1.0f : 0.0f;  // mask[a][r]
            else par_sm[jl] = 0.0f;
        }
        __syncthreads();

        const int jw = warp * 16;
        const float par0 = par_sm[jw + g];
        const float par1 = par_sm[jw + g + 8];
        const bool rpad0 = (j0 + jw + g)     >= N;
        const bool rpad1 = (j0 + jw + g + 8) >= N;

        stage_cols(k_lo, 0);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        float Ug[D / 8][4];
        #pragma unroll
        for (int nt = 0; nt < D / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) Ug[nt][e] = 0.0f;
        }

    if constexpr(PREFETCH) {
        if(j0==j_lo && qi+1<HG) {
            for(int dv=tid*8;dv<D;dv+=blockDim.x*8) {
                cp_async16(nextQ+dv,Xa_bf+anc_off+(int64_t)N*D+dv);
                cp_async16(nextDY+dv,gYa_bf+anc_off+(int64_t)N*D+dv);
            }
        }
    }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            // Prefetch k0+1; the closing barrier publishes it and frees `cur`.
            // SINGLE_COL's caller-guaranteed single trip never reuses the
            // buffer, so there is nothing to prefetch.
            int nxt = cur;
            if constexpr (!SINGLE_COL) {
                nxt = cur ^ 1;
                if (k0 + BK < k_hi) stage_cols(k0 + BK, nxt);
            }
            const bf16* xc_cur  = xc_sm + (CACHE==5 ? k0-cache_lo : cur*BK)*DPAD;
            const bf16* vc_cur  = vc_sm + (CACHE==5 ? k0-cache_lo : cur*BK)*DPAD;

            // 16 cols at a time: score GEMMs, elementwise, output GEMM.
            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], ada[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        ax[nt][e] = 0.0f; ada[nt][e] = 0.0f;
                    }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A1[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_sm + roff);
                    ldmatrix_x4(A1, a1_sm + roff);
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(ada[0], A1, bfr);
                    mma_bf16_m16n8k16(ada[1], A1, bfr + 2);
                }

                // Elementwise: weights from forward stats, Jacobian-corrected
                // grad_A; repack C-fragments as output-GEMM A-fragments.
                uint32_t gAf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;   // col within the staged tile
                    const float ilac0 = ilac_sm[cur * BK + cl];
                    const float ilac1 = ilac_sm[cur * BK + cl + 1];
                    float gA[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        // Clamping the exponent at 0 is what lets every gate be
                        // a plain multiply: live cells always have x <= m so the
                        // clamp never touches them, while dead ones can no longer
                        // reach inf and turn 0*inf into NaN. That kills the pad
                        // test too (pad rows/cols carry a zero inv-l and zero mask
                        // factors). The anchor's mask factors ride in via ilac
                        // (col) and par (row).
                        const float x = ax[hf][e];
                        const float ilac = c1 ? ilac1 : ilac0;
                        const float par  = hi ? par1 : par0;
                        const float Pa = __expf(fminf(x - ma,  0.0f)) * ilac * par;
                        gA[e] = (ada[hf][e] - sa) * Pa;
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                }

                // Output GEMM: contract cols (B fragments via ldmatrix.trans).
                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < D / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np],     gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                }
            }

            // With SINGLE_COL the next row block's stage_cols() overwrites
            // this buffer, so the barrier is still needed; only the wait_all
            // and the buffer toggle are dead.
            if constexpr (!SINGLE_COL) {
                asm volatile("cp.async.wait_all;\n" ::);
                cur = nxt;
            }
            __syncthreads();
        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows ----
        // Raw rows come from rawR when cached, else global (L2-hot); pad
        // rows contribute zeros.
        float ng[D / 4];
        const int64_t r0 = rows_off + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = rows_off + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f)
                       : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < D / 8; nt++) {
            const float2 x0 = ld2(CACHE ? rawR+((CACHE==1 ? 0 : j0-cache_lo)+jw+g)*DPAD+2*tig+nt*8 : Xr_bf+r0+nt*8, rpad0),  x1 = ld2(CACHE ? rawR+((CACHE==1 ? 0 : j0-cache_lo)+jw+g+8)*DPAD+2*tig+nt*8 : Xr_bf+r1+nt*8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < D / 4; e++) ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
        }
        if (lane < 4) {
            float* wo = wOut + warp * D;
            #pragma unroll
            for (int nt = 0; nt < D / 8; nt++) {
                wo[nt * 8 + 2 * lane]         = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1]     = ng[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid; t < D; t += blockDim.x) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; w++) acc += wOut[w * D + t];
            redOut[t] += acc;
        }
    }
    __syncthreads();

    // ---- direct stores: this CTA exclusively owns row a of dQ ----
    for (int t = tid; t < D; t += blockDim.x) gradXa[anc_off + t] = scale * redOut[t];
    if constexpr(PREFETCH) asm volatile("cp.async.wait_all;" ::);
    __syncthreads();
    } // sequential query group
#endif  // __CUDA_ARCH__ >= 800
}


// Heads in a CTA share only the raw R/Vr/S/Vs values; scores, normalizers,
// derivatives and outputs stay private to each head.
template<int W, int BK, int CACHE, int HG, bool SINGLE>
inline cudaError_t launch(const bf16* q,const bf16* dy,const bf16* r,const bf16* vr,
 const bf16* s,const bf16* vs,const float* m,const float* l,const float* delta,
 float* dq,const bool* mask,int B,int H,int N,int win,float scale,cudaStream_t stream,const uint8_t* support_meta) {
 constexpr int BJ=W*16;
 constexpr bool PREFETCH=(CACHE==5 && HG>1);
 int cb=SINGLE?1:2;
 int cap=((((win+max(BJ,BK)-1)/max(BJ,BK))*max(BJ,BK)+15)/16)*16;
 size_t shared=btc_smem_bytes(128,W,BK,SINGLE);
 if constexpr(CACHE==1) shared+=sizeof(bf16)*136*BJ;
 if constexpr(CACHE==5) shared+=sizeof(bf16)*136*(2*(cap-cb*BK)+2*cap);
 if constexpr(PREFETCH) shared+=sizeof(bf16)*256;
 auto fn=Bwd_gather_tc<W,BK,SINGLE,CACHE,HG>;
 cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared));
 if(e!=cudaSuccess)return e;
 fn<<<dim3(N,H/HG,B),W*32,shared,stream>>>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,H,N,win,scale,1,support_meta);
 return cudaGetLastError();
}
// Every schedule writes exact zero dQ for queries with at most one visible key,
// where the general kernel leaves a bf16 cancellation residue. The w16
// schedules use BJ=16, which changes the fp32 summation order.
inline bool launch_retained_dq(const bf16* q,const bf16* dy,const bf16* r,const bf16* vr,
 const bf16* s,const bf16* vs,const float* m,const float* l,const float* delta,
 float* dq,const bool* mask,int B,int H,int N,int win,float scale,cudaStream_t stream,const uint8_t* support_meta) {

 cudaError_t e;
 if(win==16) {
  if(H%4==0 && ((H>=64 && N>=512)||(H>=128 && N>=256)))
   e=launch<1,16,5,4,true>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
  else e=launch<1,16,1,1,true>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
 } else if(win==32 && H>=16 && H%4==0 && N>=128) {
  const int64_t tasks=(int64_t)B*H*N;
  // Largest head group that still leaves about one resident wave (512 CTAs).
  if(H%32==0 && tasks>=16384) e=launch<2,32,5,32,true>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
  else if(H%16==0 && tasks>=8192) e=launch<2,32,5,16,true>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
  else if(H%8==0 && tasks>=4096) e=launch<2,32,5,8,true>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
  else e=launch<2,32,5,4,true>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
 } else if(win==64 && H>=16 && H%8==0 && N>=256)
  e=launch<2,32,5,8,false>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
 else if(win==32) e=launch<2,32,0,1,true>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
 else if(win==64 || win==128) e=launch<2,32,0,1,false>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
 else return false;
 if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
 return true;
}

} // namespace sg_retained_dq
