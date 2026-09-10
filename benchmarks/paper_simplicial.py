"""Fast and Simplex (Roy et al., 2025, arXiv:2507.02754v1) Appendix B/C Triton
kernels, reconstructed from the paper's HTML listings, plus host wrappers.

Source: https://arxiv.org/html/2507.02754v1 -- Listing 1 (forward), Listing 2
(general backward, kv1 side + atomic dQ), Listing 3 (backward specialized for
small w2, no atomics). Kernel bodies are the paper's text; the paper does not
print the host code, so the wrappers below are ours, modelled on the launch in
the FBGEMM `simplicial` package (commit e568038b).

Adaptations from the printed text, all recorded:
  * the `@triton.autotune` decorators (Listing 1: BLOCK_SIZE_Q=64/BLOCK_SIZE_KV=32;
    Listing 3: BLOCK_SIZE_Q=32/BLOCK_SIZE_KV2=64; both num_warps=4, num_stages=1)
    are removed and the same constants are passed explicitly at launch, so the
    benchmark controls and records them. The handoff's fixed initial forward
    configuration is BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64, 4 warps, 1 stage.
  * K2_BIAS = V2_BIAS = 0.0 (the operator under comparison has no biases).
  * Listing 3 stores fp32 accumulators through dQ/dK2/dV2 pointers without a
    cast; the wrapper gives it fp32 buffers and converts to bf16 afterwards.
  * Layout is the paper's [b, s, k, h] (batch, seq, heads, head_dim).
Everything else -- the bf16 cast points, tf32/ieee dot precisions, the
-1e38 masking, exp in natural base, M = m + log(l) -- is as printed.
"""
import torch
import triton
import triton.language as tl


# ----- Listing 1: forward (verbatim, autotune decorator removed) -----
@triton.jit
def two_simplicial_attn_fwd_kernel(
    Q_ptr,  #
    K1_ptr,  #
    K2_ptr,  #
    V1_ptr,  #
    V2_ptr,  #
    O_ptr,  #
    M_ptr,  #
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1: tl.constexpr,
    w2: tl.constexpr,
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_k,
    out_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    num_stages: tl.constexpr,
):
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    q_start = tl.program_id(0) * BLOCK_SIZE_Q
    q_end = q_start + BLOCK_SIZE_Q
    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k

    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk
    O_ptr += qkv_offs_bk
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k

    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype) - float("inf")
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=compute_dtype)
    acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=compute_dtype)

    q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    q_mask_s = q_offs_s < seq_len
    qkv_mask_h = qkv_offs_h < head_dim
    q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
    q_mask = q_mask_s[:, None] & (qkv_mask_h[None, :])

    q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(
        compute_dtype
    )  #
    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)

    for kv1_idx in tl.range(tl.maximum(0, q_start - w1), tl.minimum(seq_len, q_end)):
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        k1_tile = (tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(compute_dtype))[
            None, :
        ]  #
        qk1 = q_tile * k1_tile  #
        qk1 = qk1.to(gemm_dtype)

        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        v1_tile = (tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(compute_dtype))[
            None, :
        ]  #

        for kv2_idx in tl.range(
            tl.maximum(0, q_start - w2),
            tl.minimum(seq_len, q_end),
            BLOCK_SIZE_KV,
            num_stages=num_stages,
        ):
            kv2_offs_s = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)
            kv2_mask_s = kv2_offs_s < seq_len
            k2t_mask = kv2_mask_s[None, :] & qkv_mask_h[:, None]
            v2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]
            k2_offs = (
                kv2_offs_s[None, :] * k2_stride_s + qkv_offs_h[:, None] * k2_stride_h
            )
            v2_offs = (
                kv2_offs_s[:, None] * v2_stride_s + qkv_offs_h[None, :] * v2_stride_h
            )
            k2t_tile = tl.load(K2_ptr + k2_offs, mask=k2t_mask).to(
                compute_dtype
            )  #
            v2_tile = tl.load(V2_ptr + v2_offs, mask=v2_mask).to(
                compute_dtype
            )  #
            k2t_tile += K2_BIAS
            v2_tile += V2_BIAS
            k2t_tile = k2t_tile.to(gemm_dtype)
            v2_tile = v2_tile.to(compute_dtype)

            qk = tl.dot(
                qk1 * softmax_scale,
                k2t_tile,
                input_precision="tf32",  #
                out_dtype=tl.float32,
            )  #

            qk_mask = q_mask_s[:, None] & kv2_mask_s[None, :]
            #
            #
            kv1_local_mask = ((q_offs_s[:, None] - w1) < kv1_idx) & (
                kv1_idx <= q_offs_s[:, None]
            )
            kv2_local_mask = ((q_offs_s[:, None] - w2) < kv2_offs_s[None, :]) & (
                kv2_offs_s[None, :] <= q_offs_s[:, None]
            )
            qk_mask &= kv1_local_mask & kv2_local_mask
            qk += tl.where(qk_mask, 0, -1.0e38)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v12_tile = v1_tile * v2_tile  #
            acc += tl.dot(
                p.to(gemm_dtype),
                v12_tile.to(gemm_dtype),
                input_precision="ieee",  #
                out_dtype=tl.float32,
            )

            m_i = m_ij
    acc = acc / l_i[:, None]

    acc = tl.where(q_mask, acc, 0.0)
    acc = acc.to(data_dtype)
    out_offs = q_offs_s[:, None] * out_stride_s + qkv_offs_h[None, :] * out_stride_h
    tl.store(O_ptr + out_offs, acc, mask=q_mask)

    m = m_i + tl.log(l_i)

    m_offs = q_offs_s * m_stride_s
    m_mask = q_offs_s < seq_len
    tl.store(M_ptr + m_offs, m, mask=m_mask)


# ----- Listing 2: general backward (verbatim) -----
@triton.jit
def two_simplicial_attn_bwd_kv1_kernel(
    Q_ptr,  #
    K1_ptr,  #
    K2_ptr,  #
    V1_ptr,  #
    V2_ptr,  #
    dO_ptr,  #
    M_ptr,  #
    D_ptr,  #
    dQ_ptr,  #
    dK1_ptr,  #
    dV1_ptr,  #
    #
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1,  #
    w2,  #
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    dq_stride_b,
    dq_stride_s,
    dq_stride_k,
    dq_stride_h,
    dk1_stride_b,
    dk1_stride_s,
    dk1_stride_k,
    dk1_stride_h,
    dv1_stride_b,
    dv1_stride_s,
    dv1_stride_k,
    dv1_stride_h,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    COMPUTE_DQ: tl.constexpr,
    num_stages: tl.constexpr,
    is_flipped: tl.constexpr,
):
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    kv1_start = tl.program_id(0) * BLOCK_SIZE_KV
    kv1_end = kv1_start + BLOCK_SIZE_KV
    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk

    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dK1_ptr += offs_b * dk1_stride_b + offs_k * dk1_stride_k
    dV1_ptr += offs_b * dv1_stride_b + offs_k * dv1_stride_k
    if COMPUTE_DQ:
        dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k

    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    qkv_mask_h = qkv_offs_h < head_dim

    kv1_offs_s = kv1_start + tl.arange(0, BLOCK_SIZE_KV)

    k1_offs = kv1_offs_s[:, None] * k1_stride_s + qkv_offs_h[None, :] * k1_stride_h
    kv1_mask_s = kv1_offs_s < seq_len
    kv1_mask = kv1_mask_s[:, None] & qkv_mask_h[None, :]
    k1_tile = tl.load(K1_ptr + k1_offs, mask=kv1_mask).to(
        compute_dtype
    )  #
    v1_offs = kv1_offs_s[:, None] * v1_stride_s + qkv_offs_h[None, :] * v1_stride_h
    v1_tile = tl.load(V1_ptr + v1_offs, mask=kv1_mask).to(
        compute_dtype
    )  #
    if is_flipped:
        k1_tile += K2_BIAS
        v1_tile += V2_BIAS
    dv1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)
    dk1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), compute_dtype)
    #
    #
    for kv2_idx in tl.range(
        tl.maximum(0, kv1_start - w2), tl.minimum(seq_len, kv1_end + w1)
    ):
        k2_offs = kv2_idx * k2_stride_s + qkv_offs_h * k2_stride_h
        k2_tile = (tl.load(K2_ptr + k2_offs, mask=qkv_mask_h).to(compute_dtype))[
            None, :
        ]  #
        v2_offs = kv2_idx * v2_stride_s + qkv_offs_h * v2_stride_h
        v2_tile = (tl.load(V2_ptr + v2_offs, mask=qkv_mask_h).to(compute_dtype))[
            None, :
        ]  #
        if not is_flipped:
            k2_tile += K2_BIAS
            v2_tile += V2_BIAS
        k1k2 = k1_tile * k2_tile  #
        v1v2 = v1_tile * v2_tile  #
        k1k2 = k1k2.to(gemm_dtype)
        v1v2 = v1v2.to(gemm_dtype)
        #
        #
        q_start = tl.maximum(kv1_start, kv2_idx)
        q_end = tl.minimum(seq_len, tl.minimum(kv1_end + w1, kv2_idx + w2))
        for q_idx in tl.range(q_start, q_end, BLOCK_SIZE_Q):
            #
            q_offs_s = q_idx + tl.arange(0, BLOCK_SIZE_Q)
            q_offs = q_offs_s[None, :] * q_stride_s + qkv_offs_h[:, None] * q_stride_h
            q_mask_s = q_offs_s < seq_len
            qt_mask = q_mask_s[None, :] & qkv_mask_h[:, None]
            qt_tile = tl.load(Q_ptr + q_offs, mask=qt_mask).to(
                gemm_dtype
            )  #
            m_offs = q_offs_s * m_stride_s
            m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(compute_dtype)[
                None, :
            ]  #
            d_offs = q_offs_s * d_stride_s
            d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(compute_dtype)[
                None, :
            ]  #
            dO_offs = (
                q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h
            )
            dO_tile = tl.load(
                dO_ptr + dO_offs, mask=q_mask_s[:, None] & qkv_mask_h[None, :]
            ).to(compute_dtype)  #
            if COMPUTE_DQ:
                dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)
            #
            #
            qkkT = tl.dot(
                k1k2, qt_tile * softmax_scale, out_dtype=tl.float32
            )  #

            #
            kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_offs_s[:, None]) & (
                kv1_offs_s[:, None] <= q_offs_s[None, :]
            )
            kv2_local_mask = ((q_offs_s - w2) < kv2_idx) & (kv2_idx <= q_offs_s)
            local_mask = (
                kv1_local_mask & kv2_local_mask[None, :]
            )  #
            qkkT = tl.where(local_mask, qkkT, -1.0e38)

            pT = tl.exp(qkkT - m_tile)  #
            pT = tl.where(local_mask, pT, 0.0)
            dOv2 = dO_tile * v2_tile  #
            dv1 += tl.dot(
                pT.to(gemm_dtype), dOv2.to(gemm_dtype), out_dtype=tl.float32
            )  #

            dpT = tl.dot(
                v1v2, tl.trans(dO_tile.to(gemm_dtype)), out_dtype=tl.float32
            )  #
            dsT = pT * (dpT - d_tile)  #
            dsT = tl.where(local_mask, dsT, 0.0)
            dsT = dsT.to(gemm_dtype)

            dk1 += (
                tl.dot(dsT, tl.trans(qt_tile), out_dtype=tl.float32)
                * k2_tile.to(tl.float32)
                * softmax_scale
            )
            if COMPUTE_DQ:
                #
                dq += (
                    tl.dot(tl.trans(dsT), k1k2, out_dtype=tl.float32) * softmax_scale
                )  #
                dq_offs = (
                    q_offs_s[:, None] * dq_stride_s + qkv_offs_h[None, :] * dq_stride_h
                )
                tl.atomic_add(
                    dQ_ptr + dq_offs, dq, mask=q_mask_s[:, None] & qkv_mask_h[None, :]
                )
    dv1_offs = kv1_offs_s[:, None] * dv1_stride_s + qkv_offs_h[None, :] * dv1_stride_h
    dk1_offs = kv1_offs_s[:, None] * dk1_stride_s + qkv_offs_h[None, :] * dk1_stride_h
    tl.store(dV1_ptr + dv1_offs, dv1.to(data_dtype), mask=kv1_mask)
    tl.store(dK1_ptr + dk1_offs, dk1.to(data_dtype), mask=kv1_mask)


# ----- Listing 3: backward specialized for small w2 (verbatim, autotune decorator removed) -----
@triton.jit
def two_simplicial_attn_bwd_kv2q_kernel(
    Q_ptr,  #
    K1_ptr,  #
    K2_ptr,  #
    V1_ptr,  #
    V2_ptr,  #
    dO_ptr,  #
    M_ptr,  #
    D_ptr,  #
    dQ_ptr,  #
    dK2_ptr,  #
    dV2_ptr,  #
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1,  #
    w2,  #
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    dq_stride_b,
    dq_stride_s,
    dq_stride_k,
    dq_stride_h,
    dk2_stride_b,
    dk2_stride_s,
    dk2_stride_k,
    dk2_stride_h,
    dv2_stride_b,
    dv2_stride_s,
    dv2_stride_k,
    dv2_stride_h,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    num_stages: tl.constexpr,
    IS_SECOND_PASS: tl.constexpr,
):
    assert BLOCK_SIZE_KV2 == BLOCK_SIZE_Q + w2
    data_dtype = tl.bfloat16
    compute_dtype = tl.float32
    gemm_dtype = tl.bfloat16

    #
    q_start = tl.program_id(0) * BLOCK_SIZE_KV2
    if IS_SECOND_PASS:
        q_start += BLOCK_SIZE_Q
    q_end = q_start + BLOCK_SIZE_Q
    kv2_start = q_start - w2

    bk = tl.program_id(1)
    offs_b = bk // num_heads
    offs_k = bk % num_heads

    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk

    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k
    dK2_ptr += offs_b * dk2_stride_b + offs_k * dk2_stride_k
    dV2_ptr += offs_b * dv2_stride_b + offs_k * dv2_stride_k

    softmax_scale = tl.cast(SM_SCALE, gemm_dtype)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    qkv_mask_h = qkv_offs_h < head_dim

    q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
    kv2_offs_s = kv2_start + tl.arange(0, BLOCK_SIZE_KV2)
    q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
    kv2_offs = kv2_offs_s[:, None] * k2_stride_s + qkv_offs_h[None, :] * k2_stride_h
    m_offs = q_offs_s * m_stride_s
    d_offs = q_offs_s * d_stride_s
    dO_offs = q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h
    q_mask_s = q_offs_s < seq_len
    q_mask = q_mask_s[:, None] & qkv_mask_h[None, :]
    kv2_mask_s = 0 <= kv2_offs_s and kv2_offs_s < seq_len
    kv2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]


    q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(
        compute_dtype
    )  #
    k2_tile = tl.load(K2_ptr + kv2_offs, mask=kv2_mask).to(gemm_dtype) #
    v2_tile = tl.load(V2_ptr + kv2_offs, mask=kv2_mask).to(gemm_dtype) #
    m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(compute_dtype) #
    d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(compute_dtype) #
    dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask).to(
        gemm_dtype
    )  #

    #
    k2_tile += K2_BIAS
    v2_tile += V2_BIAS
    k2_tile = k2_tile.to(gemm_dtype)
    v2_tile = v2_tile.to(gemm_dtype)

    dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)
    dk2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)
    dv2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)

    kv1_start = tl.maximum(0, q_start - w1)
    kv1_end = tl.minimum(seq_len, q_end)
    for kv1_idx in tl.range(kv1_start, kv1_end, num_stages=num_stages):
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        k1_tile = tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(
            compute_dtype
        )  #

        v1_tile = tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(
            compute_dtype
        )  #

        qk1_s = q_tile * (k1_tile[None, :] * softmax_scale) #
        qk1_s = qk1_s.to(gemm_dtype)
        #
        qkkT = tl.dot(k2_tile, qk1_s.T, out_dtype=tl.float32) #

        qkT_mask = kv2_mask_s[:, None] & q_mask_s[None, :]
        kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_idx) & (
            kv1_idx <= q_offs_s[None, :]
        )  #
        kv2_local_mask = ((q_offs_s[None, :] - w2) < kv2_offs_s[:, None]) & (
            kv2_offs_s[:, None] <= q_offs_s[None, :]
        )  #
        local_mask = (
            kv1_local_mask & kv2_local_mask
        )  #
        qkT_mask &= kv1_local_mask & kv2_local_mask

        pT = tl.exp(qkkT - m_tile[None, :]) #
        pT = tl.where(qkT_mask, pT, 0.0)

        qkkT = tl.where(local_mask, qkkT, -1.0e38)

        dOv1 = dO_tile * v1_tile[None, :] #
        dOv1 = dOv1.to(gemm_dtype)
        #
        dv2 += tl.dot(pT.to(gemm_dtype), dOv1, out_dtype=tl.float32)

        #
        dpT = tl.dot(v2_tile, dOv1.T, out_dtype=tl.float32)
        dsT = pT * (dpT - d_tile[None, :]) #
        dsT = tl.where(qkT_mask, dsT, 0.0)
        dsT = dsT.to(gemm_dtype) #

        #
        dk2 += tl.dot(dsT, qk1_s, out_dtype=tl.float32)

        k1k2 = k1_tile[None, :] * k2_tile #
        k1k2 = k1k2.to(gemm_dtype)

        dq += tl.dot(dsT.T, k1k2) #

    #
    if IS_SECOND_PASS:
        #load,
        prev_dk2 = tl.load(dK2_ptr + kv2_offs, kv2_mask)
        prev_dv2 = tl.load(dV2_ptr + kv2_offs, kv2_mask)
        dk2 += prev_dk2
        dv2 += prev_dv2

    dq *= softmax_scale
    tl.store(dK2_ptr + kv2_offs, dk2, kv2_mask)
    tl.store(dV2_ptr + kv2_offs, dv2, kv2_mask)
    tl.store(dQ_ptr + q_offs, dq, q_mask)


# ---------------------------------------------------------------------------
# Host wrappers (ours)
# ---------------------------------------------------------------------------

def _strides(t):
    return t.stride(0), t.stride(1), t.stride(2), t.stride(3)


def paper_fwd(q, k1, k2, v1, v2, w1, w2, BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64,
              num_warps=4, num_stages=1):
    """q, k1, k2, v1, v2: bf16 [b, s, k, h]. Returns (o bf16 [b, s, k, h],
    m fp32 [b, k, s]) with m = natural-log LSE of the scaled logits."""
    bs, seq_len, num_heads, head_dim = q.shape
    o = torch.empty_like(q)
    m = torch.empty((bs, num_heads, seq_len), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_Q), bs * num_heads)
    two_simplicial_attn_fwd_kernel[grid](
        q, k1, k2, v1, v2, o, m, bs, seq_len, num_heads, head_dim, w1, w2,
        *_strides(q), *_strides(k1), *_strides(k2), *_strides(v1), *_strides(v2),
        *_strides(o), m.stride(0), m.stride(1), m.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_KV=BLOCK_SIZE_KV, HEAD_DIM=head_dim,
        INPUT_PRECISION="tf32", SM_SCALE=head_dim ** -0.5, K2_BIAS=0.0, V2_BIAS=0.0,
        num_stages=num_stages, num_warps=num_warps)
    return o, m


def paper_delta(o, do):
    """D = rowsum(dO o O) in fp32, [b, s, k]; passed to the kernels through its
    strides so no transpose is needed."""
    return (do.float() * o.float()).sum(-1)


def _launch_kv1(q, k1, k2, v1, v2, do, m, d, dq, dk1, dv1, w1, w2, compute_dq,
                is_flipped, BLOCK_SIZE_Q, BLOCK_SIZE_KV, num_warps, num_stages):
    bs, seq_len, num_heads, head_dim = q.shape
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_KV), bs * num_heads)
    dq_t = dq if dq is not None else dk1   # unused when COMPUTE_DQ is False
    two_simplicial_attn_bwd_kv1_kernel[grid](
        q, k1, k2, v1, v2, do, m, d, dq_t, dk1, dv1,
        bs, seq_len, num_heads, head_dim, w1, w2,
        *_strides(q), *_strides(k1), *_strides(k2), *_strides(v1), *_strides(v2),
        *_strides(do), m.stride(0), m.stride(1), m.stride(2),
        d.stride(0), d.stride(2), d.stride(1),
        *_strides(dq_t), *_strides(dk1), *_strides(dv1),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_KV=BLOCK_SIZE_KV, HEAD_DIM=head_dim,
        SM_SCALE=head_dim ** -0.5, K2_BIAS=0.0, V2_BIAS=0.0, COMPUTE_DQ=compute_dq,
        num_stages=num_stages, is_flipped=is_flipped, num_warps=num_warps)


def paper_bwd_general(q, k1, k2, v1, v2, o, m, do, w1, w2, BLOCK_SIZE_Q=32,
                      BLOCK_SIZE_KV=64, num_warps=4, num_stages=1):
    """Listing 2 twice: K1/V1 with atomic dQ, then K2/V2 with the roles and
    windows swapped (is_flipped). Returns bf16 (dq, dk1, dk2, dv1, dv2)."""
    d = paper_delta(o, do)
    dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dk1, dv1, dk2, dv2 = (torch.empty_like(k1) for _ in range(4))
    _launch_kv1(q, k1, k2, v1, v2, do, m, d, dq, dk1, dv1, w1, w2, True, False,
                BLOCK_SIZE_Q, BLOCK_SIZE_KV, num_warps, num_stages)
    _launch_kv1(q, k2, k1, v2, v1, do, m, d, None, dk2, dv2, w2, w1, False, True,
                BLOCK_SIZE_Q, BLOCK_SIZE_KV, num_warps, num_stages)
    return dq.to(torch.bfloat16), dk1, dk2, dv1, dv2


def paper_bwd_small_w2(q, k1, k2, v1, v2, o, m, do, w1, w2, BLOCK_SIZE_Q=32,
                       BLOCK_SIZE_KV=64, num_warps=4, num_stages=1):
    """Listing 3 (two passes, no atomics) for dQ/dK2/dV2 plus Listing 2 without
    dQ for dK1/dV1. Requires BLOCK_SIZE_KV2 = BLOCK_SIZE_Q + w2 (w2 = 32 here)."""
    bs, seq_len, num_heads, head_dim = q.shape
    BLOCK_SIZE_KV2 = BLOCK_SIZE_Q + w2
    d = paper_delta(o, do)
    dk1, dv1 = torch.empty_like(k1), torch.empty_like(v1)
    _launch_kv1(q, k1, k2, v1, v2, do, m, d, None, dk1, dv1, w1, w2, False, False,
                BLOCK_SIZE_Q, BLOCK_SIZE_KV, num_warps, num_stages)
    dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    dk2 = torch.zeros(k2.shape, dtype=torch.float32, device=q.device)
    dv2 = torch.zeros(v2.shape, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_KV2), bs * num_heads)
    for second in (False, True):
        two_simplicial_attn_bwd_kv2q_kernel[grid](
            q, k1, k2, v1, v2, do, m, d, dq, dk2, dv2,
            bs, seq_len, num_heads, head_dim, w1, w2,
            *_strides(q), *_strides(k1), *_strides(k2), *_strides(v1), *_strides(v2),
            *_strides(do), m.stride(0), m.stride(1), m.stride(2),
            d.stride(0), d.stride(2), d.stride(1),
            *_strides(dq), *_strides(dk2), *_strides(dv2),
            BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_KV2=BLOCK_SIZE_KV2, HEAD_DIM=head_dim,
            SM_SCALE=head_dim ** -0.5, K2_BIAS=0.0, V2_BIAS=0.0,
            num_stages=num_stages, IS_SECOND_PASS=second, num_warps=num_warps)
    return dq.to(torch.bfloat16), dk1, dk2.to(torch.bfloat16), dv1, dv2.to(torch.bfloat16)
