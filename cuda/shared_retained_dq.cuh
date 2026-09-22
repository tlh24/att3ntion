#pragma once
#include "common.cuh"
#include <stdexcept>
namespace sg_retained_dq {
enum BwdRole { BWD_ALL = 0, BWD_QUERY_ANCHOR = 1, BWD_QUERY_ROWS = 2 };

// single_col drops the second column buffer: the caller promises win <= bk,
// so there is exactly one column tile.
__host__ __device__ constexpr size_t btc_smem_bytes(int D, int warps, int bk, bool masked, int role = BWD_ALL, bool single_col = false) {
    const int bj = warps * 16, dpad = D + 8;
    const int pa = role != BWD_QUERY_ROWS, pr = role != BWD_QUERY_ANCHOR, pc = role == BWD_ALL;
    const int a_tiles = 1 + pa + pr + pc, c_tiles = 2 + pc, outs = 1 + pr;
    const int cb = single_col ? 1 : 2;
    size_t b = sizeof(bf16) * ((size_t)a_tiles * bj * dpad + (size_t)cb * c_tiles * bk * dpad)
             + sizeof(float) * ((size_t)3 * D + cb * 3 * bk + 3 * bj + warps * outs * D + outs * D);
    if (masked) {
        b += sizeof(uint32_t) * cb * (size_t)(pr * bj + pc * bk * (bj / 32))
           + sizeof(float) * (cb * (size_t)bk + bj) * pa;
    }
    return b;
}

// backward.cu's Bwd_gather_tc extended for the shared-KV dQ pass (D=128,
// Hkv=1): QG queries x HG heads per CTA, run sequentially. CACHE selects the
// raw operands kept in shared memory: 0 none (epilogue re-reads rows from
// global), 1 the current row tile, 2 the R/Vr window, 3 R/Vr/S/Vs with S/Vs
// copied into the col buffers, 4 the R window, 5 R/Vr/S/Vs with S/Vs read in
// place, 6 S/Vs in place plus the row tile as in 1. PREFETCH loads the next
// head's Q/dY anchor row during the current head. SINGLETON writes exact zero
// dQ for queries with at most one visible key.
template<int D_CONST, bool MASKED, int WARPS, int BK, int ROLE = BWD_ALL, int DHT = 64, bool SINGLE_COL = false, int QG = 1, int CACHE = 0, int HG = 1, bool PREFETCH = false, bool SINGLETON = false>
__global__ __launch_bounds__(WARPS * 32, 1)
void Bwd_gather_tc(
    const bf16* __restrict__ Xa_bf,  // anchor side [B,H,N,D]
    const bf16* __restrict__ Va_bf,
    const bf16* __restrict__ gYa_bf,
    const bf16* __restrict__ Xr_bf,  // row side
    const bf16* __restrict__ Vr_bf,
    const bf16* __restrict__ gYr_bf,
    const bf16* __restrict__ Xc_bf,  // col side (streamed per k tile)
    const bf16* __restrict__ Vc_bf,
    const bf16* __restrict__ gYc_bf,
    const float* __restrict__ m_a, const float* __restrict__ l_a, const float* __restrict__ sum_a,
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,
    const float* __restrict__ m_c, const float* __restrict__ l_c, const float* __restrict__ sum_c,
    float* __restrict__ gradXa,      // [B,H,N,D] fp32, direct store
    float* __restrict__ gradVa,      // [B,H,N,D] fp32, direct store
    const bool* __restrict__ mask,   // [B,N,N] or null (MASKED only)
    int H, int N, int win, float scale, int Hkv, const uint8_t* support_meta)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(D_CONST == 64 || D_CONST == 128, "Bwd_gather_tc supports D=64/128");
    static_assert(DHT == 64 || DHT == D_CONST, "output slice is 64 channels or all of D");
    constexpr int D = D_CONST;
    constexpr int DPAD = D + 8;
    constexpr int BJ = WARPS * 16;
    constexpr int MRW = BJ / 32;    // words per col of a transposed mask window
    constexpr int KS = D / 16;      // score GEMM k-steps (D contracted)
    // The output accumulators cover DH cols per pass over the col side; at
    // D=128, DHT=64 makes two passes, recomputing the scores, rather than
    // doubling the accumulator registers. DHT=128 is one pass.
    constexpr int DH = DHT;
    constexpr int NPASS = D / DH;
    constexpr bool PA = ROLE != BWD_QUERY_ROWS;    // anchor-normalized softmax
    constexpr bool PR = ROLE != BWD_QUERY_ANCHOR;  // row-normalized softmax
    constexpr bool PC = ROLE == BWD_ALL;           // col-normalized softmax
    constexpr int NOUT = PR ? 2 : 1;               // gradXa (+ gradVa)

    const int first_a = blockIdx.x * QG;
    int query_support=0;
    for (int qi=0; qi<QG*HG && first_a+qi%QG<N; ++qi) {
    const int a = first_a + qi%QG;
    const int h = blockIdx.y*HG+qi/QG;
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

    const int COLCAP = CACHE>=5 ? ((((win+max(BJ,BK)-1)/max(BJ,BK))*max(BJ,BK)+QG-1+15)/16)*16 : CB*BK;
    extern __shared__ char smem_raw[];
    // The four A operands, anchor already folded in: scale*Xa o Xr, gYa o Vr,
    // Va o gYr, Va o Vr. Staged per row block.
    bf16* a0_sm   = reinterpret_cast<bf16*>(smem_raw);            // [BJ][DPAD]
    bf16* a1_sm   = a0_sm + BJ * DPAD;                            // PA
    bf16* a2_sm   = a1_sm + (PA ? BJ * DPAD : 0);                 // PR
    bf16* a3_sm   = a2_sm + (PR ? BJ * DPAD : 0);                 // PC
    bf16* xc_sm   = a3_sm + (PC ? BJ * DPAD : 0);                 // [CB][BK][DPAD]
    bf16* vc_sm   = xc_sm + COLCAP * DPAD;
    bf16* gyc_sm  = vc_sm + COLCAP * DPAD;                       // PC
    float* anchX  = reinterpret_cast<float*>(gyc_sm + (PC ? CB * BK * DPAD : 0));  // [D] scale*Xa
    float* anchV  = anchX + D;                                    // [D]
    float* anchG  = anchV + D;                                    // [D]
    float* mc_sm  = anchG + D;                                    // [CB][BK]
    float* ilc_sm = mc_sm + CB * BK;
    float* sc_sm  = ilc_sm + CB * BK;
    float* mr_sm  = sc_sm + CB * BK;                             // [BJ]
    float* ilr_sm = mr_sm + BJ;
    float* sr_sm  = ilr_sm + BJ;
    float* wOut   = sr_sm + BJ;                                  // [WARPS][NOUT*D]
    float* redOut = wOut + WARPS * NOUT * D;                     // [NOUT*D]
    // MASKED only (host omits these bytes): the tile in flight's mask windows.
    uint32_t* msk_sm  = reinterpret_cast<uint32_t*>(redOut + NOUT * D);   // [CB][BJ] PR
    uint32_t* mskT_sm = msk_sm + (PR ? CB * BJ : 0);                     // [CB][BK][MRW] PC
    float* ilac_sm = reinterpret_cast<float*>(mskT_sm + (PC ? CB * BK * MRW : 0));  // [CB][BK] PA
    float* par_sm  = ilac_sm + CB * BK;                                  // [BJ] PA

    // Offsets by operand role. Query-head operands (Q, dY, the forward stats and
    // every output, including the per-head gradient partials) live at (b*H + h);
    // key/value operands at the KV head (b*Hkv + h / (H/Hkv)), identical unless
    // the caller shares KV across query heads. With the query as anchor the
    // anchor side is query-indexed and rows/cols are KV; with the queries as rows
    // the anchor (R or S) is KV, the rows (Q, dY) are query-indexed, cols are KV.
    const int64_t bh = (int64_t)b * H + h;
    const int64_t kvh = (int64_t)b * Hkv + h / (H / Hkv);
    const int64_t q_off = bh * N * D, kv_off = kvh * N * D;
    const int64_t anc_off  = (ROLE == BWD_QUERY_ROWS ? kv_off : q_off) + (int64_t)a * D;  // Xa / Va / gYa reads
    const int64_t a_off    = q_off + (int64_t)a * D;                                      // gradXa / gradVa stores
    const int64_t rows_off = (ROLE == BWD_QUERY_ROWS ? q_off : kv_off);                   // Xr / Vr / gYr
    const int64_t nd_off   = kv_off;                                                      // Xc / Vc / gYc
    const int64_t st_off = bh * N;

    if constexpr(SINGLETON) {
        static_assert(QG==1,"singleton fast path uses one query");
        if(qi==0) {
            if(support_meta) query_support=support_meta[(int64_t)b*N+a];
            else {
            int count=0;
            for(int k0=max(0,a-win+1);k0<=a;k0+=32) {
                int key=k0+lane;
                unsigned bits=__ballot_sync(0xffffffffu,key<=a && key<N && mask[((int64_t)b*N+a)*N+key]);
                count+=__popc(bits);
            }
            query_support=count;
            }
        }
        if(query_support<=1) {
            for(int d=tid;d<D;d+=blockDim.x) gradXa[a_off+d]=0.0f;
            continue;
        }
    }
    constexpr int DV = D / 8;
    const int cache_lo = max(0, first_a - win + 1);
    const int cache_len = ((((win + max(BJ,BK)-1)/max(BJ,BK))*max(BJ,BK)+QG-1+15)/16)*16;
    bf16* rawR = reinterpret_cast<bf16*>(smem_raw + btc_smem_bytes(D,WARPS,BK,MASKED,ROLE,SINGLE_COL) + 2*(COLCAP-CB*BK)*DPAD*sizeof(bf16));
    bf16* rawV = rawR + cache_len * DPAD;
    bf16* rawS = CACHE>=5 ? xc_sm : rawV+cache_len*DPAD;
    bf16* rawVS = CACHE>=5 ? vc_sm : rawS+cache_len*DPAD;
    if constexpr (CACHE >= 2) {
        if(qi==0) {
            for(int idx=tid;idx<cache_len*DV;idx+=blockDim.x) {
                const int r=idx/DV,d=(idx%DV)*8,j=cache_lo+r;
                if(j<N) {
                    if constexpr(CACHE!=6) cp_async16(rawR+r*DPAD+d,Xr_bf+rows_off+(int64_t)j*D+d);
                    if constexpr(CACHE!=4 && CACHE!=6) cp_async16(rawV+r*DPAD+d,Vr_bf+rows_off+(int64_t)j*D+d);
                    if constexpr(CACHE==3 || CACHE>=5) {
                        cp_async16(rawS+r*DPAD+d,Xc_bf+nd_off+(int64_t)j*D+d);
                        cp_async16(rawVS+r*DPAD+d,Vc_bf+nd_off+(int64_t)j*D+d);
                    }
                } else {
                    uint4 z=make_uint4(0,0,0,0);
                    if constexpr(CACHE!=6) *reinterpret_cast<uint4*>(rawR+r*DPAD+d)=z;
                    if constexpr(CACHE!=4 && CACHE!=6) *reinterpret_cast<uint4*>(rawV+r*DPAD+d)=z;
                    if constexpr(CACHE==3 || CACHE>=5) {
                        *reinterpret_cast<uint4*>(rawS+r*DPAD+d)=z;
                        *reinterpret_cast<uint4*>(rawVS+r*DPAD+d)=z;
                    }
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
        if constexpr (PR || PC) anchV[d] = bf2f(Va_bf[anc_off + d]);
        if constexpr (PA) anchG[d] = bf2f(PREFETCH && qi>0 ? nextDY[d] : gYa_bf[anc_off+d]);
    }
    for (int d = tid; d < NOUT * D; d += blockDim.x) redOut[d] = 0.0f;

    const float ma  = PA ? m_a[st_off + a] : 0.0f;
    const float ila = PA ? 1.0f / fmaxf(l_a[st_off + a], DENOM_EPS) : 0.0f;
    const float sa  = PA ? sum_a[st_off + a] : 0.0f;
    const bool* mb  = MASKED ? mask + (int64_t)b * N * N : nullptr;

    // ---- row blocks of BJ rows, one 16-row tile per warp ----
    // win > 0 restricts the row blocks and, per block, the col tiles to those
    // that can hold visible pairs (sg_row_bounds / sg_col_bounds, keyed on
    // which side of the pass the query sits); the mask still decides every
    // cell.
    constexpr int SIDE = (ROLE == BWD_QUERY_ROWS) ? SG_QUERY_ROWS : SG_QUERY_ANCHOR;
    int j_lo, j_hi;
    sg_row_bounds(SIDE, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();  // previous iteration's smem reads (and anchor) done
        int k_lo, k_hi;
        sg_col_bounds(SIDE, a, j0, BJ, N, win, BK, k_lo, k_hi);

        // Stage col tile k0 into buffer `buf`: matrices, forward stats and
        // (masked) mask windows. Zero-filled pads carry a zero inv-l and zero
        // mask bits, gating the tail tile off with no per-cell test.
        auto stage_cols = [&](int k0, int buf) {
            bf16* xs = xc_sm + buf * BK * DPAD;
            bf16* vs = vc_sm + buf * BK * DPAD;
            bf16* gs = gyc_sm + buf * BK * DPAD;
            if constexpr(CACHE < 5) { for (int idx = tid; idx < BK * DV; idx += blockDim.x) {
                const int kl = idx / DV, dv = (idx % DV) * 8;
                const int k = k0 + kl;
                if (k < N) {
                    const int64_t off = nd_off + (int64_t)k * D + dv;
                    if constexpr(CACHE==3) *reinterpret_cast<uint4*>(xs+kl*DPAD+dv)=*reinterpret_cast<uint4*>(rawS+(k-cache_lo)*DPAD+dv);
                    else cp_async16(xs + kl * DPAD + dv, Xc_bf + off);
                    if constexpr(CACHE==3) *reinterpret_cast<uint4*>(vs+kl*DPAD+dv)=*reinterpret_cast<uint4*>(rawVS+(k-cache_lo)*DPAD+dv);
                    else cp_async16(vs + kl * DPAD + dv, Vc_bf + off);
                    if constexpr (PC) cp_async16(gs + kl * DPAD + dv, gYc_bf + off);
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(xs + kl * DPAD + dv) = z;
                    *reinterpret_cast<uint4*>(vs + kl * DPAD + dv) = z;
                    if constexpr (PC) *reinterpret_cast<uint4*>(gs + kl * DPAD + dv) = z;
                }
            }
            }
            for (int kl = tid; kl < BK; kl += blockDim.x) {
                const int k = k0 + kl;
                float mc = 0.0f, ilc = 0.0f, sc = 0.0f;
                if (PC && k < N) {
                    mc  = m_c[st_off + k];
                    ilc = 1.0f / fmaxf(l_c[st_off + k], DENOM_EPS);
                    sc  = sum_c[st_off + k];
                }
                if constexpr (MASKED) {
                    // Separable column factors, free per cell: mask[c][a]
                    // zeroes P_c's inv-l, mask[a][c] rides in ilac_sm.
                    if (PC && (k >= N || !mb[(int64_t)k * N + a])) ilc = 0.0f;
                    if constexpr (PA) {
                        ilac_sm[buf * BK + kl] =
                            (k < N && mb[(int64_t)a * N + k]) ? ila : 0.0f;
                    }
                }
                mc_sm[buf * BK + kl]  = mc;
                ilc_sm[buf * BK + kl] = ilc;
                sc_sm[buf * BK + kl]  = sc;
            }
            if constexpr (MASKED && PR) {
                for (int jl = tid; jl < BJ; jl += blockDim.x) {
                    const int j = j0 + jl;
                    msk_sm[buf * BJ + jl] = (j < N)
                        ? sg_pack_mask32(mb + (int64_t)j * N + k0, min(BK, N - k0))
                        : 0u;
                }
            }
            if constexpr (MASKED && PC) {
                for (int idx = tid; idx < BK * MRW; idx += blockDim.x) {
                    const int kl = idx / MRW, w = idx - kl * MRW;
                    const int k = k0 + kl, jb = j0 + w * 32;
                    mskT_sm[buf * BK * MRW + idx] = (k < N && jb < N)
                        ? sg_pack_mask32(mb + (int64_t)k * N + jb, min(32, N - jb))
                        : 0u;
                }
            }
        };

        // A operands: the anchor rescale is done in fp32 with a single bf16
        // rounding, matching Y_gather_tc's Qp precision. Pad rows stage as
        // zeros.
        for (int idx = tid; idx < BJ * DV; idx += blockDim.x) {
            const int jl = idx / DV, dv = (idx % DV) * 8;
            const int j = j0 + jl;
            uint4 xq = make_uint4(0, 0, 0, 0), vq = xq, gq = xq;
            if (j < N) {
                const int64_t off = rows_off + (int64_t)j * D + dv;
                xq = *reinterpret_cast<const uint4*>(CACHE>=2 && CACHE!=6 ? rawR+(j-cache_lo)*DPAD+dv : Xr_bf+off);
                if constexpr (PA || PC) vq = *reinterpret_cast<const uint4*>(CACHE>=2 && CACHE!=4 && CACHE!=6 ? rawV+(j-cache_lo)*DPAD+dv : Vr_bf+off);
                if constexpr (PR) gq = *reinterpret_cast<const uint4*>(gYr_bf + off);
            }
            if constexpr(CACHE==1 || CACHE==6) *reinterpret_cast<uint4*>(rawR+jl*DPAD+dv)=xq;
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xq);
            const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vq);
            const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gq);
            uint4 a0, a1, a2, a3;
            __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&a0);
            __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(&a1);
            __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&a2);
            __nv_bfloat162* p3 = reinterpret_cast<__nv_bfloat162*>(&a3);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int d = dv + 2 * e;
                const float2 x = __bfloat1622float2(xp[e]);
                const float2 v = __bfloat1622float2(vp[e]);
                const float2 gy = __bfloat1622float2(gp[e]);
                p0[e] = __floats2bfloat162_rn(x.x * anchX[d], x.y * anchX[d + 1]);
                p1[e] = __floats2bfloat162_rn(v.x * anchG[d], v.y * anchG[d + 1]);
                p2[e] = __floats2bfloat162_rn(gy.x * anchV[d], gy.y * anchV[d + 1]);
                p3[e] = __floats2bfloat162_rn(v.x * anchV[d], v.y * anchV[d + 1]);
            }
            *reinterpret_cast<uint4*>(a0_sm + jl * DPAD + dv) = a0;
            if constexpr (PA) *reinterpret_cast<uint4*>(a1_sm + jl * DPAD + dv) = a1;
            if constexpr (PR) *reinterpret_cast<uint4*>(a2_sm + jl * DPAD + dv) = a2;
            if constexpr (PC) *reinterpret_cast<uint4*>(a3_sm + jl * DPAD + dv) = a3;
        }
        for (int jl = tid; jl < BJ; jl += blockDim.x) {
            const int j = j0 + jl;
            if (j < N) {
                float il = 0.0f;
                if constexpr (PR) {
                    mr_sm[jl] = m_r[st_off + j];
                    il        = 1.0f / fmaxf(l_r[st_off + j], DENOM_EPS);
                    sr_sm[jl] = sum_r[st_off + j];
                }
                if constexpr (MASKED) {
                    // Both row-side factors are staged rather than kept in
                    // registers: a live register costs more than a
                    // rematerializable smem read.
                    if (PR && !mb[(int64_t)j * N + a]) il = 0.0f;          // mask[r][a]
                    if constexpr (PA) par_sm[jl] = mb[(int64_t)a * N + j] ? 1.0f : 0.0f;  // mask[a][r]
                }
                ilr_sm[jl] = il;
            } else {
                mr_sm[jl] = 0.0f; ilr_sm[jl] = 0.0f; sr_sm[jl] = 0.0f;
                if constexpr (MASKED && PA) par_sm[jl] = 0.0f;
            }
        }
        __syncthreads();

        const int jw = warp * 16;
        const float mr0 = PR ? mr_sm[jw + g]     : 0.0f, sr0 = PR ? sr_sm[jw + g]     : 0.0f;
        const float mr1 = PR ? mr_sm[jw + g + 8] : 0.0f, sr1 = PR ? sr_sm[jw + g + 8] : 0.0f;
        const float ilr0 = PR ? ilr_sm[jw + g] : 0.0f, ilr1 = PR ? ilr_sm[jw + g + 8] : 0.0f;
        const float par0 = (MASKED && PA) ? par_sm[jw + g]     : 1.0f;
        const float par1 = (MASKED && PA) ? par_sm[jw + g + 8] : 1.0f;
        // Zero-filled pad rows produce x = 0, which with extreme stats can
        // push exp(x - m)/l past bf16 range (inf -> 0*inf = NaN downstream).
        // Forcing pad cells to NEG_INF zeroes all three weights exactly.
        const bool rpad0 = (j0 + jw + g)     >= N;
        const bool rpad1 = (j0 + jw + g + 8) >= N;

        // The two row-side mask factors were folded into ilr_sm / par_sm during
        // staging above. Pad rows pack to zero bits in both windows, so no
        // per-cell pad test is needed. Rows jw+g and jw+g+8 always share one
        // word of the transposed window (a 16-row tile never straddles a 32-row
        // boundary), so one load covers both at shifts 0 and 8.
        const int rwd = (jw + g) >> 5;
        const int rb  = (jw + g) & 31;

        for (int pass = 0; pass < NPASS; pass++) {
            const int d0 = pass * DH;   // this pass's output col slice
            stage_cols(k_lo, 0);
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();

            float Ug[DH / 8][4], U1[PR ? DH / 8 : 1][4], U2[PC ? DH / 8 : 1][4];
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    Ug[nt][e] = 0.0f;
                    if constexpr (PR) U1[nt][e] = 0.0f;
                    if constexpr (PC) U2[nt][e] = 0.0f;
                }
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
                const bf16* xc_cur  = xc_sm + (CACHE>=5 ? k0-cache_lo : cur*BK)*DPAD;
                const bf16* vc_cur  = vc_sm + (CACHE>=5 ? k0-cache_lo : cur*BK)*DPAD;
                const bf16* gyc_cur = gyc_sm + cur * BK * DPAD;
                const float* mc_cur  = mc_sm + cur * BK;
                const float* ilc_cur = ilc_sm + cur * BK;
                const float* sc_cur  = sc_sm + cur * BK;

                // 16 cols at a time: score GEMMs, elementwise, output GEMMs.
                #pragma unroll
                for (int s2 = 0; s2 < BK / 16; s2++) {
                    float ax[2][4], ada[2][4], adr[2][4], adc[2][4];
                    #pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            ax[nt][e] = 0.0f; ada[nt][e] = 0.0f; adr[nt][e] = 0.0f; adc[nt][e] = 0.0f;
                        }
                    }
                    const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                    const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                    const bf16* bg = gyc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                    #pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        uint32_t A0[4], A1[4], A2[4], A3[4], bfr[4];
                        const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                        ldmatrix_x4(A0, a0_sm + roff);
                        if constexpr (PA) ldmatrix_x4(A1, a1_sm + roff);
                        if constexpr (PR) ldmatrix_x4(A2, a2_sm + roff);
                        if constexpr (PC) ldmatrix_x4(A3, a3_sm + roff);
                        ldmatrix_x4(bfr, bx + ks * 16);
                        mma_bf16_m16n8k16(ax[0], A0, bfr);
                        mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                        ldmatrix_x4(bfr, bv + ks * 16);   // shared by d_a and d_r
                        if constexpr (PA) {
                            mma_bf16_m16n8k16(ada[0], A1, bfr);
                            mma_bf16_m16n8k16(ada[1], A1, bfr + 2);
                        }
                        if constexpr (PR) {
                            mma_bf16_m16n8k16(adr[0], A2, bfr);
                            mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                        }
                        if constexpr (PC) {
                            ldmatrix_x4(bfr, bg + ks * 16);
                            mma_bf16_m16n8k16(adc[0], A3, bfr);
                            mma_bf16_m16n8k16(adc[1], A3, bfr + 2);
                        }
                    }

                    // Elementwise: weights from forward stats, Jacobian-corrected
                    // grad_A; repack C-fragments as output-GEMM A-fragments.
                    uint32_t gAf[4], Prf[4], Pcf[4];
                    #pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        const int cl = s2 * 16 + hf * 8 + 2 * tig;   // col within the staged tile
                        const int c  = k0 + cl;
                        const float mc0 = PC ? mc_cur[cl]     : 0.0f, ilc0 = PC ? ilc_cur[cl]     : 0.0f;
                        const float mc1 = PC ? mc_cur[cl + 1] : 0.0f, ilc1 = PC ? ilc_cur[cl + 1] : 0.0f;
                        const float sc0 = PC ? sc_cur[cl] : 0.0f, sc1 = PC ? sc_cur[cl + 1] : 0.0f;
                        // Only the two 2-D factors are left per cell, one aligned load
                        // per pair in either window: bits 0/1 of rc* are mask[r][c] and
                        // mask[r][c+1] (c even, so the pair shares a word), bits 0/8 of
                        // cr* are mask[c][r] and mask[c][r+8]. So rc* indexes by row and
                        // cr* by col: the windows' packing axes, swapped.
                        uint32_t rc0 = ~0u, rc1 = ~0u, cr0 = ~0u, cr1 = ~0u;
                        float ilac0 = 0.0f, ilac1 = 0.0f;
                        if constexpr (MASKED && PR) {
                            const uint32_t* mw  = msk_sm + cur * BJ;
                            rc0 = mw[jw + g]     >> cl;
                            rc1 = mw[jw + g + 8] >> cl;
                        }
                        if constexpr (MASKED && PC) {
                            const uint32_t* mtw = mskT_sm + cur * BK * MRW;
                            cr0 = mtw[cl * MRW + rwd]       >> rb;
                            cr1 = mtw[(cl + 1) * MRW + rwd] >> rb;
                        }
                        if constexpr (MASKED && PA) {
                            ilac0 = ilac_sm[cur * BK + cl];
                            ilac1 = ilac_sm[cur * BK + cl + 1];
                        }
                        float gA[4], Pr[4], Pc[4];
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            const bool hi = (e >= 2), c1 = (e & 1);
                            const float mrr = hi ? mr1 : mr0;
                            const float ilr = hi ? ilr1 : ilr0;
                            const float srr = hi ? sr1 : sr0;
                            const float mcc = c1 ? mc1 : mc0;
                            const float ilc = c1 ? ilc1 : ilc0;
                            const float scc = c1 ? sc1 : sc0;
                            float Pa = 0.0f;
                            Pr[e] = 0.0f; Pc[e] = 0.0f;
                            if constexpr (MASKED) {
                                // Clamping the exponent at 0 is what lets every gate be
                                // a plain multiply or select: live cells always have
                                // x <= m so the clamp never touches them, while dead
                                // ones can no longer reach inf and turn 0*inf into NaN.
                                // That kills the pad test too (pad rows/cols carry a
                                // zero inv-l and zero mask bits), so no `pad` term
                                // appears below. Remaining per cell: mask[r][c] and
                                // mask[c][r]; the anchor's own factors already rode in
                                // via ilac (col) and par (row).
                                const float x = ax[hf][e];
                                const float ilac = c1 ? ilac1 : ilac0;
                                const float par  = hi ? par1 : par0;
                                // Gate the *scale*, never the exp. Guarding the whole
                                // expression lets nvcc branch around the MUFU, and a
                                // scattered mask then diverges inside the warp and runs
                                // both sides. Selecting on inv-l keeps every cell's cost
                                // identical.
                                const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                                const float ilcg = (((c1 ? cr1 : cr0) >> (hi ? 8 : 0)) & 1u)
                                                 ? ilc : 0.0f;
                                if constexpr (PA) Pa    = __expf(fminf(x - ma,  0.0f)) * ilac * par;
                                if constexpr (PR) Pr[e] = __expf(fminf(x - mrr, 0.0f)) * ilrg;
                                if constexpr (PC) Pc[e] = __expf(fminf(x - mcc, 0.0f)) * ilcg;
                            } else {
                                const bool pad = (hi ? rpad1 : rpad0) | (c + c1 >= N);
                                const float x = pad ? NEG_INF : ax[hf][e];
                                if constexpr (PA) Pa    = __expf(x - ma)  * ila;
                                if constexpr (PR) Pr[e] = __expf(x - mrr) * ilr;
                                if constexpr (PC) Pc[e] = __expf(x - mcc) * ilc;
                            }
                            if constexpr (ROLE == BWD_ALL) {
                                gA[e] = (ada[hf][e] - sa) * Pa
                                      + (adr[hf][e] - srr) * Pr[e]
                                      + (adc[hf][e] - scc) * Pc[e];
                            } else if constexpr (PA) {
                                gA[e] = (ada[hf][e] - sa) * Pa;
                            } else {
                                gA[e] = (adr[hf][e] - srr) * Pr[e];
                            }
                        }
                        gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                        gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                        if constexpr (PR) {
                            Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                            Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                        }
                        if constexpr (PC) {
                            Pcf[2 * hf + 0] = pack_bf162(Pc[0], Pc[1]);
                            Pcf[2 * hf + 1] = pack_bf162(Pc[2], Pc[3]);
                        }
                    }

                    // Output GEMMs: contract cols (B fragments via ldmatrix.trans).
                    const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8 + d0;
                    const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8 + d0;
                    const bf16* pg = gyc_cur + (s2 * 16 + lrow) * DPAD + lcol8 + d0;
                    #pragma unroll
                    for (int np = 0; np < DH / 16; np++) {
                        uint32_t bfr[4];
                        ldmatrix_x4_trans(bfr, px + np * 16);
                        mma_bf16_m16n8k16(Ug[2 * np],     gAf, bfr);
                        mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                        if constexpr (PR) {
                            ldmatrix_x4_trans(bfr, pv + np * 16);
                            mma_bf16_m16n8k16(U1[2 * np],     Prf, bfr);
                            mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                        }
                        if constexpr (PC) {
                            ldmatrix_x4_trans(bfr, pg + np * 16);
                            mma_bf16_m16n8k16(U2[2 * np],     Pcf, bfr);
                            mma_bf16_m16n8k16(U2[2 * np + 1], Pcf, bfr + 2);
                        }
                    }
                }

                // With SINGLE_COL the next pass's stage_cols() overwrites this
                // buffer, so the barrier is still needed; only the wait_all and
                // the buffer toggle are dead.
                if constexpr (!SINGLE_COL) {
                    asm volatile("cp.async.wait_all;\n" ::);
                    cur = nxt;
                }
                __syncthreads();
            }

            // ---- epilogue: Hadamard row-collapse of this warp's 16 rows ----
            // Raw rows come from rawR when cached, else global (L2-hot); pad
            // rows contribute zeros.
            float ng[DH / 4], nv[PR ? DH / 4 : 1];
            const int64_t r0 = rows_off + (int64_t)min(j0 + jw + g, N - 1) * D + d0 + 2 * tig;
            const int64_t r1 = rows_off + (int64_t)min(j0 + jw + g + 8, N - 1) * D + d0 + 2 * tig;
            auto ld2 = [](const bf16* p, bool pad) {
                return pad ? make_float2(0.0f, 0.0f)
                           : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
            };
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                const float2 x0 = ld2(CACHE ? rawR+(((CACHE==1 || CACHE==6) ? 0 : j0-cache_lo)+jw+g)*DPAD+d0+2*tig+nt*8 : Xr_bf+r0+nt*8, rpad0),  x1 = ld2(CACHE ? rawR+(((CACHE==1 || CACHE==6) ? 0 : j0-cache_lo)+jw+g+8)*DPAD+d0+2*tig+nt*8 : Xr_bf+r1+nt*8, rpad1);
                ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
                ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
                if constexpr (PC) {
                    const float2 v0 = ld2(Vr_bf + r0 + nt * 8, rpad0),  v1 = ld2(Vr_bf + r1 + nt * 8, rpad1);
                    const float2 g0 = ld2(gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(gYr_bf + r1 + nt * 8, rpad1);
                    nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2]
                                   + v0.x * U2[nt][0] + v1.x * U2[nt][2];
                    nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3]
                                   + v0.y * U2[nt][1] + v1.y * U2[nt][3];
                } else if constexpr (PR) {
                    const float2 g0 = ld2(gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(gYr_bf + r1 + nt * 8, rpad1);
                    nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
                    nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
                }
            }
            #pragma unroll
            for (int off = 4; off <= 16; off <<= 1) {
                #pragma unroll
                for (int e = 0; e < DH / 4; e++) {
                    ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                    if constexpr (PR) nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
                }
            }
            if (lane < 4) {
                float* wo = wOut + warp * NOUT * D + d0;
                #pragma unroll
                for (int nt = 0; nt < DH / 8; nt++) {
                    wo[nt * 8 + 2 * lane]         = ng[2 * nt + 0];
                    wo[nt * 8 + 2 * lane + 1]     = ng[2 * nt + 1];
                    if constexpr (PR) {
                        wo[D + nt * 8 + 2 * lane]     = nv[2 * nt + 0];
                        wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
                    }
                }
            }
        }
        __syncthreads();
        for (int t = tid; t < NOUT * D; t += blockDim.x) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; w++) acc += wOut[w * NOUT * D + t];
            redOut[t] += acc;
        }
    }
    __syncthreads();

    // ---- direct stores: this CTA exclusively owns row a of both outputs ----
    for (int t = tid; t < D; t += blockDim.x) {
        gradXa[a_off + t] = scale * redOut[t];
        if constexpr (PR) gradVa[a_off + t] = redOut[D + t];
    }
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
 float* dq,const bool* mask,int B,int H,int N,int win,float scale,cudaStream_t stream,const uint8_t* support_meta=nullptr) {
 constexpr int QG=1, BJ=W*16;
 constexpr bool PREFETCH=(CACHE==5 && HG>1);
 int cb=SINGLE?1:2;
 int cap=((((win+max(BJ,BK)-1)/max(BJ,BK))*max(BJ,BK)+15)/16)*16;
 size_t shared=btc_smem_bytes(128,W,BK,true,BWD_QUERY_ANCHOR,SINGLE);
 if constexpr(CACHE==1) shared+=sizeof(bf16)*136*BJ;
 if constexpr(CACHE==5) shared+=sizeof(bf16)*136*(2*(cap-cb*BK)+2*cap);
 if constexpr(PREFETCH) shared+=sizeof(bf16)*256;
 auto fn=Bwd_gather_tc<128,true,W,BK,BWD_QUERY_ANCHOR,128,SINGLE,QG,CACHE,HG,PREFETCH,true>;
 cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared));
 if(e!=cudaSuccess)return e;
 fn<<<dim3(N,H/HG,B),W*32,shared,stream>>>(q,nullptr,dy,r,vr,nullptr,s,vs,nullptr,
  m,l,delta,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,dq,nullptr,mask,H,N,win,scale,1,support_meta);
 return cudaGetLastError();
}
// Every schedule writes exact zero dQ for queries with at most one visible key,
// where the general kernel leaves a bf16 cancellation residue. The w16
// schedules use BJ=16, which changes the fp32 summation order.
inline bool launch_retained_dq(const bf16* q,const bf16* dy,const bf16* r,const bf16* vr,
 const bf16* s,const bf16* vs,const float* m,const float* l,const float* delta,
 float* dq,const bool* mask,int B,int H,int N,int win,float scale,cudaStream_t stream,const uint8_t* support_meta=nullptr) {

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
