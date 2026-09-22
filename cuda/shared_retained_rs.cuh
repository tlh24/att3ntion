#pragma once
// Independently derived raw-window residency family; exact masks and separate contractions.
#include "common.cuh"
namespace att3_shared_rs {
__device__ __forceinline__ uint32_t load_packed_mask(const uint32_t* row,int words,int k) {
 const int word=k>>5,shift=k&31;
 const uint32_t lo=word<words?row[word]:0u;
 const uint32_t hi=(shift && word+1<words)?row[word+1]:0u;
 return ((lo>>shift)|(shift?(hi<<(32-shift)):0u))&0xffffu;
}

template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w16_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=32;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int h0 = blockIdx.y * HG;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * CAP * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        anchV[d] = bf2f((d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel]);
    }
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;

    const int raw_lo=max(0,a-win+1);
    for(int idx=tid;idx<CAP*DV;idx+=NTHR) {
        const int kl=idx/DV,dv=(idx%DV)*8,k=raw_lo+kl;
        bf16* xs=xc_sm+kl*DPAD+dv;
        bf16* vs=vc_sm+kl*DPAD+dv;
        bf16* xs1=xs+CAP*DPAD;
        bf16* vs1=vs+CAP*DPAD;
        if(k<N) {
            const int64_t off=kv_off+(int64_t)k*D+dv;
            cp_async16(xs,Xc_bf+off);cp_async16(vs,Vc_bf+off);
            cp_async16(xs1,Xa_bf+off);cp_async16(vs1,Va_bf+off);
        } else {
            const uint4 z=make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(xs)=z;*reinterpret_cast<uint4*>(vs)=z;
            *reinterpret_cast<uint4*>(xs1)=z;*reinterpret_cast<uint4*>(vs1)=z;
        }
    }
    asm volatile("cp.async.wait_all;\n" ::);
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;

            const bf16* xc_cur = xc_sm + (role * CAP + k0-raw_lo) * DPAD;
            const bf16* vc_cur = vc_sm + (role * CAP + k0-raw_lo) * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));
                        const float2 ay=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A2[ae]));
                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        A2[ae]=pack_bf162(ay.x*anchV[role*D+ad],ay.y*anchV[role*D+ad+1]);
                    }
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w16(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w16_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w16_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}



template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w32_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=64;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int h0 = blockIdx.y * HG;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * CAP * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        reinterpret_cast<bf16*>(anchV)[d] = (d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel];
    }
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;

    const int raw_lo=max(0,a-win+1);
    for(int idx=tid;idx<CAP*DV;idx+=NTHR) {
        const int kl=idx/DV,dv=(idx%DV)*8,k=raw_lo+kl;
        bf16* xs=xc_sm+kl*DPAD+dv;
        bf16* vs=vc_sm+kl*DPAD+dv;
        bf16* xs1=xs+CAP*DPAD;
        bf16* vs1=vs+CAP*DPAD;
        if(k<N) {
            const int64_t off=kv_off+(int64_t)k*D+dv;
            cp_async16(xs,Xc_bf+off);cp_async16(vs,Vc_bf+off);
            cp_async16(xs1,Xa_bf+off);cp_async16(vs1,Va_bf+off);
        } else {
            const uint4 z=make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(xs)=z;*reinterpret_cast<uint4*>(vs)=z;
            *reinterpret_cast<uint4*>(xs1)=z;*reinterpret_cast<uint4*>(vs1)=z;
        }
    }
    asm volatile("cp.async.wait_all;\n" ::);
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;

            const bf16* xc_cur = xc_sm + (role * CAP + k0-raw_lo) * DPAD;
            const bf16* vc_cur = vc_sm + (role * CAP + k0-raw_lo) * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));

                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        const __nv_bfloat162 vv=*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(anchV)+role*D+ad);
                        const __nv_bfloat162 prod=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&A2[ae]),vv);
                        A2[ae]=*reinterpret_cast<const uint32_t*>(&prod);
                    }
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w32(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w32_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w32_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}




template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w64_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=80;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int h0 = blockIdx.y * HG;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * CAP * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        reinterpret_cast<bf16*>(anchV)[d] = (d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel];
    }
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;

    const int raw_lo=max(0,a-win+1);
    int raw_hi=raw_lo;
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);
        const int fill_lo=raw_hi;
        raw_hi=max(raw_hi,k_hi);
        for(int idx=tid;idx<(raw_hi-fill_lo)*DV;idx+=NTHR) {
            const int kl=idx/DV,dv=(idx%DV)*8,k=fill_lo+kl;
            const int unwrapped=k-raw_lo;
            const int slot=unwrapped>=CAP?unwrapped-CAP:unwrapped;
            bf16* xs=xc_sm+slot*DPAD+dv;
            bf16* vs=vc_sm+slot*DPAD+dv;
            bf16* xs1=xs+CAP*DPAD;
            bf16* vs1=vs+CAP*DPAD;
            if(k<N) {
                const int64_t off=kv_off+(int64_t)k*D+dv;
                cp_async16(xs,Xc_bf+off);cp_async16(vs,Vc_bf+off);
                cp_async16(xs1,Xa_bf+off);cp_async16(vs1,Va_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(xs)=z;*reinterpret_cast<uint4*>(vs)=z;
                *reinterpret_cast<uint4*>(xs1)=z;*reinterpret_cast<uint4*>(vs1)=z;
            }
        }


        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;

            const bf16* xc_cur = xc_sm + role * CAP * DPAD;
            const bf16* vc_cur = vc_sm + role * CAP * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const int br_unwrapped=k0-raw_lo+s2*16+brow;
                const int br=br_unwrapped>=CAP?br_unwrapped-CAP:br_unwrapped;
                const bf16* bx = xc_cur + br * DPAD + bcol8;
                const bf16* bv = vc_cur + br * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));

                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        const __nv_bfloat162 vv=*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(anchV)+role*D+ad);
                        const __nv_bfloat162 prod=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&A2[ae]),vv);
                        A2[ae]=*reinterpret_cast<const uint32_t*>(&prod);
                    }
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const int pr_unwrapped=k0-raw_lo+s2*16+lrow;
                const int pr=pr_unwrapped>=CAP?pr_unwrapped-CAP:pr_unwrapped;
                const bf16* px = xc_cur + pr * DPAD + lcol8;
                const bf16* pv = vc_cur + pr * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = redOut + (size_t)lh * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] += ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] += ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] += nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] += nv[2 * nt + 1];
            }
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w64(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w64_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w64_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}




template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w16_v2_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=32;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int head_base = blockIdx.y * HG * 2;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * CAP * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;

    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        anchV[d] = bf2f((d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel]);
    }


    const int raw_lo=max(0,a-win+1);
    for(int idx=tid;idx<CAP*DV;idx+=NTHR) {
        const int kl=idx/DV,dv=(idx%DV)*8,k=raw_lo+kl;
        bf16* xs=xc_sm+kl*DPAD+dv;
        bf16* vs=vc_sm+kl*DPAD+dv;
        bf16* xs1=xs+CAP*DPAD;
        bf16* vs1=vs+CAP*DPAD;
        if(k<N) {
            const int64_t off=kv_off+(int64_t)k*D+dv;
            cp_async16(xs,Xc_bf+off);cp_async16(vs,Vc_bf+off);
            cp_async16(xs1,Xa_bf+off);cp_async16(vs1,Va_bf+off);
        } else {
            const uint4 z=make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(xs)=z;*reinterpret_cast<uint4*>(vs)=z;
            *reinterpret_cast<uint4*>(xs1)=z;*reinterpret_cast<uint4*>(vs1)=z;
        }
    }
    asm volatile("cp.async.wait_all;\n" ::);
    #pragma unroll 1
    for(int visit=0;visit<2;visit++) {
    const int h0=head_base+visit*HG;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;

            const bf16* xc_cur = xc_sm + (role * CAP + k0-raw_lo) * DPAD;
            const bf16* vc_cur = vc_sm + (role * CAP + k0-raw_lo) * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));
                        const float2 ay=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A2[ae]));
                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        A2[ae]=pack_bf162(ay.x*anchV[role*D+ad],ay.y*anchV[role*D+ad+1]);
                    }
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
    __syncthreads();
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w16_v2(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w16_v2_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w16_v2_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}



template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w16_v4_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=32;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int head_base = blockIdx.y * HG * 4;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * CAP * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;

    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        anchV[d] = bf2f((d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel]);
    }


    const int raw_lo=max(0,a-win+1);
    for(int idx=tid;idx<CAP*DV;idx+=NTHR) {
        const int kl=idx/DV,dv=(idx%DV)*8,k=raw_lo+kl;
        bf16* xs=xc_sm+kl*DPAD+dv;
        bf16* vs=vc_sm+kl*DPAD+dv;
        bf16* xs1=xs+CAP*DPAD;
        bf16* vs1=vs+CAP*DPAD;
        if(k<N) {
            const int64_t off=kv_off+(int64_t)k*D+dv;
            cp_async16(xs,Xc_bf+off);cp_async16(vs,Vc_bf+off);
            cp_async16(xs1,Xa_bf+off);cp_async16(vs1,Va_bf+off);
        } else {
            const uint4 z=make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(xs)=z;*reinterpret_cast<uint4*>(vs)=z;
            *reinterpret_cast<uint4*>(xs1)=z;*reinterpret_cast<uint4*>(vs1)=z;
        }
    }
    asm volatile("cp.async.wait_all;\n" ::);
    #pragma unroll 1
    for(int visit=0;visit<4;visit++) {
    const int h0=head_base+visit*HG;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;

            const bf16* xc_cur = xc_sm + (role * CAP + k0-raw_lo) * DPAD;
            const bf16* vc_cur = vc_sm + (role * CAP + k0-raw_lo) * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));
                        const float2 ay=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A2[ae]));
                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        A2[ae]=pack_bf162(ay.x*anchV[role*D+ad],ay.y*anchV[role*D+ad+1]);
                    }
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
    __syncthreads();
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w16_v4(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w16_v4_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w16_v4_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}



template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w32_v2_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=64;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int head_base = blockIdx.y * HG * 2;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * CAP * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;

    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        reinterpret_cast<bf16*>(anchV)[d] = (d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel];
    }


    const int raw_lo=max(0,a-win+1);
    for(int idx=tid;idx<CAP*DV;idx+=NTHR) {
        const int kl=idx/DV,dv=(idx%DV)*8,k=raw_lo+kl;
        bf16* xs=xc_sm+kl*DPAD+dv;
        bf16* vs=vc_sm+kl*DPAD+dv;
        bf16* xs1=xs+CAP*DPAD;
        bf16* vs1=vs+CAP*DPAD;
        if(k<N) {
            const int64_t off=kv_off+(int64_t)k*D+dv;
            cp_async16(xs,Xc_bf+off);cp_async16(vs,Vc_bf+off);
            cp_async16(xs1,Xa_bf+off);cp_async16(vs1,Va_bf+off);
        } else {
            const uint4 z=make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(xs)=z;*reinterpret_cast<uint4*>(vs)=z;
            *reinterpret_cast<uint4*>(xs1)=z;*reinterpret_cast<uint4*>(vs1)=z;
        }
    }
    asm volatile("cp.async.wait_all;\n" ::);
    #pragma unroll 1
    for(int visit=0;visit<2;visit++) {
    const int h0=head_base+visit*HG;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;

            const bf16* xc_cur = xc_sm + (role * CAP + k0-raw_lo) * DPAD;
            const bf16* vc_cur = vc_sm + (role * CAP + k0-raw_lo) * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));

                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        const __nv_bfloat162 vv=*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(anchV)+role*D+ad);
                        const __nv_bfloat162 prod=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&A2[ae]),vv);
                        A2[ae]=*reinterpret_cast<const uint32_t*>(&prod);
                    }
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
    __syncthreads();
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w32_v2(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w32_v2_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w32_v2_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}




template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w32_v4_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=64;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int head_base = blockIdx.y * HG * 4;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * CAP * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;

    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        reinterpret_cast<bf16*>(anchV)[d] = (d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel];
    }


    const int raw_lo=max(0,a-win+1);
    for(int idx=tid;idx<CAP*DV;idx+=NTHR) {
        const int kl=idx/DV,dv=(idx%DV)*8,k=raw_lo+kl;
        bf16* xs=xc_sm+kl*DPAD+dv;
        bf16* vs=vc_sm+kl*DPAD+dv;
        bf16* xs1=xs+CAP*DPAD;
        bf16* vs1=vs+CAP*DPAD;
        if(k<N) {
            const int64_t off=kv_off+(int64_t)k*D+dv;
            cp_async16(xs,Xc_bf+off);cp_async16(vs,Vc_bf+off);
            cp_async16(xs1,Xa_bf+off);cp_async16(vs1,Va_bf+off);
        } else {
            const uint4 z=make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(xs)=z;*reinterpret_cast<uint4*>(vs)=z;
            *reinterpret_cast<uint4*>(xs1)=z;*reinterpret_cast<uint4*>(vs1)=z;
        }
    }
    asm volatile("cp.async.wait_all;\n" ::);
    #pragma unroll 1
    for(int visit=0;visit<4;visit++) {
    const int h0=head_base+visit*HG;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+dv;
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;

            const bf16* xc_cur = xc_sm + (role * CAP + k0-raw_lo) * DPAD;
            const bf16* vc_cur = vc_sm + (role * CAP + k0-raw_lo) * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));

                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        const __nv_bfloat162 vv=*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(anchV)+role*D+ad);
                        const __nv_bfloat162 prod=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&A2[ae]),vv);
                        A2[ae]=*reinterpret_cast<const uint32_t*>(&prod);
                    }
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
    __syncthreads();
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w32_v4(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w32_v4_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w32_v4_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}




template<int G, int WPH, int BK, bool RAW=false, bool SPECIAL=false>
__device__ __forceinline__
void Bwd_rows_w128_impl(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 128, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32, CAP=144, VB=2;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int h0 = blockIdx.y * HG;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rawQ_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * CAP * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * VB * BK * DPAD);   // [D]
    float* anchV = anchX + 2 * D;                                   // [D]
    float* mr_sm = reinterpret_cast<float*>(reinterpret_cast<bf16*>(anchV)+2*D);                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        reinterpret_cast<bf16*>(anchV)[d] = (d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel];
    }
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;

    const int raw_lo=max(0,a-win+1);
    int raw_hi=raw_lo;
    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);
        const int fill_lo=raw_hi;
        raw_hi=max(raw_hi,k_hi);
        for(int idx=tid;idx<(raw_hi-fill_lo)*DV;idx+=NTHR) {
            const int kl=idx/DV,dv=(idx%DV)*8,k=fill_lo+kl;
            const int unwrapped=k-raw_lo;
            const int slot=unwrapped>=CAP?unwrapped-CAP:unwrapped;
            bf16* xs=xc_sm+slot*DPAD+(dv^((slot&7)*8));

            bf16* xs1=xs+CAP*DPAD;

            if(k<N) {
                const int64_t off=kv_off+(int64_t)k*D+dv;
                cp_async16(xs,Xc_bf+off);
                cp_async16(xs1,Xa_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(xs)=z;
                *reinterpret_cast<uint4*>(xs1)=z;
            }
        }


        auto stage_values = [&](int k0,int buf) {
            for(int idx=tid;idx<BK*DV;idx+=NTHR) {
                const int kl=idx/DV,dv=(idx%DV)*8,k=k0+kl;
                bf16* vs=vc_sm+(buf*BK+kl)*DPAD+(dv^((kl&7)*8));
                bf16* vs1=vs+VB*BK*DPAD;
                if(k<N) {
                    const int64_t off=kv_off+(int64_t)k*D+dv;
                    cp_async16(vs,Vc_bf+off);cp_async16(vs1,Va_bf+off);
                } else {
                    const uint4 z=make_uint4(0,0,0,0);
                    *reinterpret_cast<uint4*>(vs)=z;*reinterpret_cast<uint4*>(vs1)=z;
                }
            }
        };
        auto stage_masks = [&]() {
            for(int idx=tid;idx<((k_hi-k_lo)/BK)*BJ;idx+=NTHR) {
                const int kt=idx/BJ,jl=idx%BJ,j=j0+jl,k0=k_lo+kt*BK;
                msk_sm[idx]=(j<N)?load_packed_mask(packed_mask+((int64_t)b*N+j)*mask_words,mask_words,k0):0u;
            }
        };

        for(int idx=tid;idx<HG*BJ*DV;idx+=NTHR) {
            const int hh=idx/(BJ*DV),jl=(idx/DV)%BJ,dv=(idx%DV)*8,j=j0+jl;
            bf16* qs=rawQ_sm+((size_t)hh*BJ+jl)*DPAD+(dv^((jl&7)*8));
            bf16* ys=rawDY_sm+((size_t)hh*BJ+jl)*DPAD+(dv^((jl&7)*8));
            if(j<N) {
                const int64_t off=(((int64_t)b*H+h0+hh)*N+j)*D+dv;
                cp_async16(qs,Xr_bf+off);cp_async16(ys,gYr_bf+off);
            } else {
                const uint4 z=make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(qs)=z;*reinterpret_cast<uint4*>(ys)=z;
            }
        }

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_values(k_lo,0);
        stage_masks();
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = rawQ_sm + (size_t)head * BJ * DPAD;
        const bf16* a2_h = rawDY_sm + (size_t)head * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool single0=SPECIAL && (j0+jw+g<N) && support[(int64_t)b*N+j0+jw+g]==1;
        const bool single1=SPECIAL && (j0+jw+g+8<N) && support[(int64_t)b*N+j0+jw+g+8]==1;
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;
            if constexpr(VB==2) { if(k0+BK<k_hi) stage_values(k0+BK,nxt); }

            const bf16* xc_cur = xc_sm + role * CAP * DPAD;
            const bf16* vc_cur = vc_sm + (role*VB+cur)*BK*DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const int br_unwrapped=k0-raw_lo+s2*16+brow;
                const int br=br_unwrapped>=CAP?br_unwrapped-CAP:br_unwrapped;
                const bf16* bx = xc_cur + br * DPAD + bcol8;
                const bf16* bv = vc_cur + brow * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ((ks * 16 + lcol8)^((lrow&7)*8));
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    #pragma unroll
                    for (int ae=0;ae<4;ae++) {
                        const int ad=ks*16 + 2*tig + (ae/2)*8;
                        const float2 aq=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&A0[ae]));

                        A0[ae]=pack_bf162(aq.x*anchX[role*D+ad],aq.y*anchX[role*D+ad+1]);
                        const __nv_bfloat162 vv=*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(anchV)+role*D+ad);
                        const __nv_bfloat162 prod=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&A2[ae]),vv);
                        A2[ae]=*reinterpret_cast<const uint32_t*>(&prod);
                    }
                    ldmatrix_x4(bfr, xc_cur+br*DPAD+((ks*16+bcol8)^((br&7)*8)));
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, vc_cur+brow*DPAD+((ks*16+bcol8)^((brow&7)*8)));
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + ((k0-k_lo)/BK) * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = (hi ? single1 : single0) ? (ilrg>0.0f ? 1.0f : 0.0f) : __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (hi ? single1 : single0) ? 0.0f : (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const int pr_unwrapped=k0-raw_lo+s2*16+lrow;
                const int pr=pr_unwrapped>=CAP?pr_unwrapped-CAP:pr_unwrapped;
                const bf16* px = xc_cur + pr * DPAD + lcol8;
                const bf16* pv = vc_cur + lrow * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, xc_cur+pr*DPAD+((np*16+lcol8)^((pr&7)*8)));
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, vc_cur+lrow*DPAD+((np*16+lcol8)^((lrow&7)*8)));
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }

            if(k0+BK<k_hi) {
                if constexpr(VB==1) { __syncthreads();stage_values(k0+BK,0); }
                asm volatile("cp.async.wait_all;\n" ::);
                __syncthreads();
                if constexpr(VB==2) cur=nxt;
            }
        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + ((2*tig+nt*8)^((g&7)*8)) : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + ((2*tig+nt*8)^((g&7)*8)) : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + ((2*tig+nt*8)^((g&7)*8)) : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + ((2*tig+nt*8)^((g&7)*8)) : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = redOut + (size_t)lh * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] += ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] += ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] += nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] += nv[2 * nt + 1];
            }
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        reinterpret_cast<bf16*>(role == 0 ? gradXa : gradXc)[out_off + t] = __float2bfloat16_rn(scale * redOut[lh * NOUT * D + t]);
        reinterpret_cast<bf16*>(role == 0 ? gradVa : gradVc)[out_off + t] = __float2bfloat16_rn(redOut[lh * NOUT * D + D + t]);
    }
#endif
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_w128(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const uint8_t* __restrict__ support, const uint32_t* __restrict__ packed_mask, int mask_words, const bool* __restrict__ mask, int H, int N, float scale, int win)
{
    const int a=blockIdx.x,b=blockIdx.z;
    bool has_singleton=false;
    for(int j=a+threadIdx.x;j<min(N,a+win);j+=blockDim.x)
        has_singleton=has_singleton || support[(int64_t)b*N+j]==1;
    const bool special=__syncthreads_or(has_singleton);
    if(special) Bwd_rows_w128_impl<G,WPH,BK,RAW,true>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
    else Bwd_rows_w128_impl<G,WPH,BK,RAW,false>(Xa_bf,Va_bf,Xr_bf,gYr_bf,Xc_bf,Vc_bf,m_r,l_r,sum_r,gradXa,gradVa,gradXc,gradVc,support,packed_mask,mask_words,mask,H,N,scale,win);
}




// Reference metadata producer. A shared mask-preparation pass can supply the
// same [B,N] bytes (0=no support,1=singleton,2=multiple) without this launch.
__global__ void prepare_row_support(const bool* mask,uint8_t* support,int N,int win) {
 const int row=blockIdx.x,b=blockIdx.y,t=threadIdx.x,k=row-win+1+t;
 const bool admitted=t<win && k>=0 && k<N && mask[((int64_t)b*N+row)*N+k];
 const int count=__syncthreads_count(admitted);
 if(t==0) support[(int64_t)b*N+row]=count>1?2:count;
}
inline void launch_row_support(const bool* mask,uint8_t* support,int B,int N,int win,cudaStream_t stream) {
 prepare_row_support<<<dim3(N,B),128,0,stream>>>(mask,support,N,win);
}

inline bool launch_retained_rs(
 const bf16* xa,const bf16* va,const bf16* xr,const bf16* dyr,
 const bf16* xc,const bf16* vc,const float* m,const float* l,const float* delta,
 float* gx,float* gv,float* gxc,float* gvc,const bool* mask,
 int B,int H,int N,int win,float scale,int max_smem_optin,cudaStream_t stream,const uint8_t* support,const uint32_t* packed_mask,int mask_words) {
 if(H%2) return false;
 // H100-only current launch heuristic: retain at least four CTA waves per SM,
 // and shorten visits near boundaries where anchor work varies most strongly.
 int visits=1;
 const int64_t work=(int64_t)B*H*N;
 if(win<=32 && H%8==0 && N>4*win && work>=4224) visits=4;
 else if(win<=32 && H%4==0 && work>=2112) visits=2;
 if(win==16 && visits==1) {
   constexpr size_t smem=63360;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w16<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w16<4,1,16,true><<<dim3(N,H/2,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
 if(win==32 && visits==1) {
   constexpr size_t smem=98304;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w32<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w32<4,1,16,true><<<dim3(N,H/2,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
 if(win==64 && visits==1) {
   constexpr size_t smem=111680;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w64<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w64<4,1,16,true><<<dim3(N,H/2,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
 if(win==16 && visits==2) {
   constexpr size_t smem=63360;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w16_v2<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w16_v2<4,1,16,true><<<dim3(N,H/4,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
 if(win==16 && visits==4) {
   constexpr size_t smem=63360;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w16_v4<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w16_v4<4,1,16,true><<<dim3(N,H/8,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
 if(win==32 && visits==2) {
   constexpr size_t smem=98304;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w32_v2<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w32_v2<4,1,16,true><<<dim3(N,H/4,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
 if(win==32 && visits==4) {
   constexpr size_t smem=98304;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w32_v4<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w32_v4<4,1,16,true><<<dim3(N,H/8,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
 if(win==128) {
   constexpr size_t smem=113472;
   if(max_smem_optin<int(smem)) return false;
   int device=0; if(cudaGetDevice(&device)!=cudaSuccess) return false;
   static thread_local int initialized_device=-1;
   if(initialized_device!=device) {
     if(cudaFuncSetAttribute(Bwd_rows_w128<4,1,16,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem))!=cudaSuccess) return false;
     initialized_device=device;
   }
   Bwd_rows_w128<4,1,16,true><<<dim3(N,H/2,B),128,smem,stream>>>(xa,va,xr,dyr,xc,vc,m,l,delta,gx,gv,gxc,gvc,support,packed_mask,mask_words,mask,H,N,scale,win);
   return true;
 }
return false;
}
} // namespace overnight_rs_v4
