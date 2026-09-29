#pragma once
#include "../common.cuh"
#include <stdexcept>
namespace sg_wide_dq {
__host__ __device__ constexpr size_t btc_smem_bytes(int D, int warps, int bk) {
    const int bj = warps * 16, dpad = D + 8;
    size_t b = sizeof(bf16) * ((size_t)2 * bj * dpad + (size_t)4 * bk * dpad)
             + sizeof(float) * ((size_t)3 * D + 2 * 3 * bk + 3 * bj + warps * D + D);
    b += sizeof(float) * (2 * (size_t)bk + bj);
    return b;
}

constexpr int WARPS = 2, BK = 32;

// backward.cu's Bwd_gather_tc extended for the shared-KV dQ pass (D=128,
// Hkv=1): HG heads per CTA, run sequentially, with the query as anchor. The
// R/Vr row tile and the S/Vs window stay in shared memory, S/Vs read in place.
// Row tiles are the outer loop (heads inner, one redOut per head), so only one
// raw row tile is kept, and only rows the window can reach are copied. The next
// head's Q/dY anchor row is prefetched during the current head, and queries
// with at most one visible key get exact zero dQ.
template<int HG>
__global__ __launch_bounds__(WARPS * 32, 1)
void Bwd_gather_tc(
    const bf16* __restrict__ Xa_bf,  // anchor side [B,H,N,D]
    const bf16* __restrict__ gYa_bf,
    const bf16* __restrict__ Xr_bf,  // row side
    const bf16* __restrict__ Vr_bf,
    const bf16* __restrict__ Xc_bf,  // col side
    const bf16* __restrict__ Vc_bf,
    const float* __restrict__ m_a, const float* __restrict__ l_a, const float* __restrict__ sum_a,
    float* __restrict__ gradXa,      // [B,H,N,D] fp32, direct store
    const bool* __restrict__ mask,   // [B,N,N]
    int H, int N, int win, float scale, int Hkv,const uint8_t* support_meta)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128;
    constexpr int DPAD = D + 8;
    constexpr int BJ = WARPS * 16;
    constexpr int KS = D / 16;      // score GEMM k-steps (D contracted)

    const int first_a = blockIdx.x;
    int query_support=0;
    const int outer_count=(min(N,first_a+1)-max(0,first_a-win+1)+BJ-1)/BJ;
    for(int outer=0;outer<outer_count;++outer) {
    const int tile_row_lo=max(0,first_a-win+1)+outer*BJ;
    for (int qi=0; qi<HG && first_a<N; ++qi) {
    const int a = first_a;
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

    const int COLCAP = ((win+BJ-1)/BJ)*BJ;
    extern __shared__ char smem_raw[];
    // The two A operands, anchor already folded in: scale*Xa o Xr, gYa o Vr.
    // Staged per row block.
    bf16* a0_sm   = reinterpret_cast<bf16*>(smem_raw);            // [BJ][DPAD]
    bf16* a1_sm   = a0_sm + BJ * DPAD;
    bf16* xc_sm   = a1_sm + BJ * DPAD;                            // S window
    bf16* vc_sm   = xc_sm + COLCAP * DPAD;                        // Vs window
    float* anchX  = reinterpret_cast<float*>(vc_sm + COLCAP * DPAD);  // [D] scale*Xa
    float* anchG  = anchX + D;                                    // [D]
    float* wOut   = anchG + D;                                    // [WARPS][D]
    float* redBase = wOut + WARPS * D;
    float* redOut=redBase+qi*D;                                   // [D]
    float* ilac_sm = redBase+HG*D;                                // [2][BK]
    float* par_sm  = ilac_sm + 2 * BK;                            // [BJ]

    // Q/dY (the anchor) and the output live at (b*H + h); R/Vr/S/Vs at the KV
    // head (b*Hkv + h / (H/Hkv)).
    const int64_t bh = (int64_t)b * H + h;
    const int64_t kvh = (int64_t)b * Hkv + h / (H / Hkv);
    const int64_t q_off = bh * N * D, kv_off = kvh * N * D;
    const int64_t anc_off  = q_off + (int64_t)a * D;   // Xa / gYa reads
    const int64_t a_off    = q_off + (int64_t)a * D;   // gradXa stores
    const int64_t rows_off = kv_off;                   // Xr / Vr
    const int64_t nd_off   = kv_off;                   // Xc / Vc
    const int64_t st_off = bh * N;

    if(qi==0 && outer==0) query_support=support_meta[(int64_t)b*N+a];
    if(query_support<=1) {
        for(int d=tid;d<D;d+=blockDim.x) gradXa[a_off+d]=0.0f;
        continue;
    }
    constexpr int DV = D / 8;
    const int cache_lo = max(0, first_a - win + 1);
    const int raw_row_lo=tile_row_lo;
    bf16* rawR = reinterpret_cast<bf16*>(smem_raw + btc_smem_bytes(D,WARPS,BK) + 2*(COLCAP-2*BK)*DPAD*sizeof(bf16) + (HG-1)*D*sizeof(float) - D*sizeof(float) - (3*2*BK+3*BJ)*sizeof(float));
    bf16* rawV = rawR + BJ*DPAD;
    bf16* rawS = xc_sm;
    bf16* rawVS = vc_sm;

    if(qi==0) {
        const int copy_rows=(outer>0 || qi>0) ? BJ : ((min(win,a+1)+BK-1)/BK)*BK;
        for(int idx=tid;idx<copy_rows*DV;idx+=blockDim.x) {
            int rr=idx/DV,d=(idx%DV)*8;
            if(rr<BJ) {
                int j=raw_row_lo+rr;
                if(j<N) {
                    cp_async16(rawR+rr*DPAD+d,Xr_bf+rows_off+(int64_t)j*D+d);
                    cp_async16(rawV+rr*DPAD+d,Vr_bf+rows_off+(int64_t)j*D+d);
                } else {
                    uint4 z=make_uint4(0,0,0,0);
                    *reinterpret_cast<uint4*>(rawR+rr*DPAD+d)=z;
                    *reinterpret_cast<uint4*>(rawV+rr*DPAD+d)=z;
                }
            }
            if(qi==0 && outer==0) {
                int j=cache_lo+rr;
                if(j<N) {
                    cp_async16(rawS+rr*DPAD+d,Xc_bf+nd_off+(int64_t)j*D+d);
                    cp_async16(rawVS+rr*DPAD+d,Vc_bf+nd_off+(int64_t)j*D+d);
                } else {
                    uint4 z=make_uint4(0,0,0,0);
                    *reinterpret_cast<uint4*>(rawS+rr*DPAD+d)=z;
                    *reinterpret_cast<uint4*>(rawVS+rr*DPAD+d)=z;
                }
            }
        }
        asm volatile("cp.async.wait_all;" ::);
        __syncthreads();
    }
    bf16* nextQ = rawV+BJ*DPAD;
    bf16* nextDY = nextQ+D;
    // ---- one-time loads: anchor only ----
    for (int d = tid; d < D; d += blockDim.x) {
        anchX[d] = scale * bf2f(qi>0 ? nextQ[d] : Xa_bf[anc_off+d]);
        anchG[d] = bf2f(qi>0 ? nextDY[d] : gYa_bf[anc_off+d]);
    }
    if(outer==0) for (int d = tid; d < D; d += blockDim.x) redOut[d] = 0.0f;

    const float ma  = m_a[st_off + a];
    const float ila = 1.0f / fmaxf(l_a[st_off + a], DENOM_EPS);
    const float sa  = sum_a[st_off + a];
    const bool* mb  = mask + (int64_t)b * N * N;

    // ---- the row block of BJ rows, one 16-row tile per warp ----
    // win > 0 restricts the col tiles to those that can hold visible pairs
    // (sg_col_bounds); the mask still decides every cell.
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ANCHOR, a, N, win, BJ, j_lo, j_hi);
    j_lo=tile_row_lo; j_hi=min(j_hi,tile_row_lo+BJ);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();  // previous iteration's smem reads (and anchor) done
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ANCHOR, a, j0, BJ, N, win, BK, k_lo, k_hi);

        // Stage col tile k0's mask factor into buffer `buf`: mask[a][c] rides in
        // ilac_sm. Pads carry a zero inv-l, gating the tail tile off with no
        // per-cell test.
        auto stage_cols = [&](int k0, int buf) {
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
                xq = *reinterpret_cast<const uint4*>(rawR+(j-raw_row_lo)*DPAD+dv);
                vq = *reinterpret_cast<const uint4*>(rawV+(j-raw_row_lo)*DPAD+dv);
            }
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

        if(j0==j_lo && qi+1<HG) {
            for(int dv=tid*8;dv<D;dv+=blockDim.x*8) {
                cp_async16(nextQ+dv,Xa_bf+anc_off+(int64_t)N*D+dv);
                cp_async16(nextDY+dv,gYa_bf+anc_off+(int64_t)N*D+dv);
            }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            // Prefetch k0+1; the closing barrier publishes it and frees `cur`.
            const int nxt = cur ^ 1;
            if (k0 + BK < k_hi) stage_cols(k0 + BK, nxt);
            const bf16* xc_cur  = xc_sm + (k0-cache_lo)*DPAD;
            const bf16* vc_cur  = vc_sm + (k0-cache_lo)*DPAD;

            // 16 cols at a time: score GEMMs, elementwise, output GEMMs.
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
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;   // col within the window tile
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
                        // test too (pad cols carry a zero inv-l), so no `pad`
                        // term appears below. The anchor's mask factors ride in
                        // via ilac (col) and par (row).
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

            asm volatile("cp.async.wait_all;\n" ::);
            cur = nxt;
            __syncthreads();
        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows ----
        // Raw rows come from rawR; pad rows contribute zeros.
        float ng[D / 4];
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f)
                       : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < D / 8; nt++) {
            const float2 x0 = ld2(rawR+(j0-raw_row_lo+jw+g)*DPAD+2*tig+nt*8, rpad0),  x1 = ld2(rawR+(j0-raw_row_lo+jw+g+8)*DPAD+2*tig+nt*8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < D / 4; e++)
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
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

    // ---- direct stores: this CTA exclusively owns row a of the output ----
    if(outer+1==outer_count) for (int t = tid; t < D; t += blockDim.x) {
        gradXa[a_off + t] = scale * redOut[t];
    }
    asm volatile("cp.async.wait_all;" ::);
    __syncthreads();
    } // sequential heads
    } // row-tile outer loop
#endif  // __CUDA_ARCH__ >= 800
}

template<int HG>
inline cudaError_t launch(const bf16* q,const bf16* dy,const bf16* r,const bf16* vr,
 const bf16* s,const bf16* vs,const float* m,const float* l,const float* delta,
 float* dq,const bool* mask,int B,int H,int N,int win,float scale,cudaStream_t stream,
 const uint8_t* support_meta) {
 int cap=((win+31)/32)*32;
 size_t shared=btc_smem_bytes(128,WARPS,BK)
  +sizeof(bf16)*136*(2*(cap-64)+64)+(HG-1)*128*sizeof(float)-1152;
 auto fn=Bwd_gather_tc<HG>;
 cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared));
 if(e!=cudaSuccess)return e;
 fn<<<dim3(N,H/HG,B),64,shared,stream>>>(q,dy,r,vr,s,vs,
  m,l,delta,dq,mask,H,N,win,scale,1,support_meta);
 return cudaGetLastError();
}
// Returns false outside the w64/w128 shapes this schedule is used for; the
// caller then falls back to launch_retained_dq.
inline bool launch_wide_dq(const bf16* q,const bf16* dy,const bf16* r,const bf16* vr,
 const bf16* s,const bf16* vs,const float* m,const float* l,const float* delta,
 float* dq,const bool* mask,int B,int H,int N,int win,float scale,cudaStream_t stream,
 const uint8_t* support_meta) {
 const int64_t tasks=(int64_t)B*H*N;
 int group=0;
 if(win==64 && H>=16 && H%4==0 && N>=128 && tasks>=4096)
  group=(H%8==0 && tasks>=32768)?8:4;
 if(win==128 && H>=64 && H%8==0 && N>=128 && tasks>=16384)
  group=N>=256?8:4;
 if(!group)return false;
 cudaError_t e=group==8
  ?launch<8>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta)
  :launch<4>(q,dy,r,vr,s,vs,m,l,delta,dq,mask,B,H,N,win,scale,stream,support_meta);
 if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
 return true;
}

}
