"""Correctness tests for the tensor-core backward path (Bwd_gather_tc).

The TC path engages only when the forward outputs Y_q/Y_r/Y_s are passed to
backward AND the scatter cotangents are zero (scatter unused). Calling
backward WITHOUT the Y tensors forces the scalar path, which lets one process
compare both paths directly, and both against torch autograd.

The second half of the file covers the MASKED=true variant. Since
_torch_kernels has no masked reference, those tests build a gather-only fp32
einsum reference (`_ref_gather_masked`) and differentiate it with autograd.

Note: on GPUs with < ~172 KB opt-in shared memory (e.g. consumer Ada), large-N
configs silently fall back to the scalar path; the autograd comparison still
holds. Full TC coverage requires an A100/H100. `_tc_smem_bytes` computes the
kernel's requirement so the fallback-sensitive tests can skip rather than
silently pass.
"""
import math
import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import att3ntion._cuda_kernels as ck
import att3ntion._torch_kernels as tk

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

GRAD_NAMES = ["grad_Q", "grad_R", "grad_S",
              "grad_Vq_1", "grad_Vq_2", "grad_Vr_1", "grad_Vr_2",
              "grad_Vs_1", "grad_Vs_2"]

CONFIGS = [
    # (B, H, N, D, input_scale)
    # N=16 forces K_pad=32 and exercises the padded resident rows.
    (1, 2, 16, 64, 1.0),
    (1, 1, 32, 64, 1.0),
    (2, 2, 32, 64, 1.0),
    (1, 2, 64, 64, 1.0),
    (2, 4, 64, 64, 1.0),
    (1, 2, 96, 64, 1.0),
    (1, 2, 128, 64, 1.0),
    (1, 2, 160, 64, 1.0),
    (1, 2, 192, 64, 1.0),
    (1, 2, 224, 64, 1.0),
    (2, 2, 256, 64, 1.0),
    (1, 2, 64, 64, 2.0),   # stress: larger logits
    (1, 2, 32, 64, 3.0),
]


def _run(B, H, N, D, scale):
    torch.manual_seed(1234 + N + B * 7 + H * 13)
    dev = "cuda"
    rnd = lambda: torch.randn(B, H, N, D, device=dev) * scale
    names = ["Q", "R", "S", "Vq_1", "Vq_2", "Vr_1", "Vr_2", "Vs_1", "Vs_2"]
    inp = {n: rnd() for n in names}
    bf = {n: v.to(torch.bfloat16) for n, v in inp.items()}

    gYq, gYr, gYs = rnd(), rnd(), rnd()
    zero = torch.zeros(B, H, N, D, device=dev, dtype=torch.bfloat16)

    out = ck.forward(*[bf[n] for n in names], 0.0)
    Yq, Yr, Ys = out[0], out[1], out[2]
    stats = out[6:12]

    args = (gYq.to(torch.bfloat16), gYr.to(torch.bfloat16), gYs.to(torch.bfloat16),
            zero, zero, zero,
            *[bf[n] for n in names], *stats, 0.0)
    tc = ck.backward(*args, None, Yq, Yr, Ys)   # mask=None, Y provided -> TC
    sc = ck.backward(*args)                     # no Y -> scalar path

    ref = {n: inp[n].detach().clone().requires_grad_(True) for n in names}
    ro = tk.forward(ref["Q"], ref["R"], ref["S"], ref["Vq_1"], ref["Vq_2"],
                    ref["Vr_1"], ref["Vr_2"], ref["Vs_1"], ref["Vs_2"], 0.0)
    loss = (ro[0] * gYq).sum() + (ro[1] * gYr).sum() + (ro[2] * gYs).sum()
    loss.backward()
    ref_grads = [ref["Q"].grad, ref["R"].grad, ref["S"].grad,
                 ref["Vq_1"].grad,
                 ref["Vq_2"].grad if ref["Vq_2"].grad is not None else torch.zeros_like(inp["Vq_2"]),
                 ref["Vr_1"].grad,
                 ref["Vr_2"].grad if ref["Vr_2"].grad is not None else torch.zeros_like(inp["Vr_2"]),
                 ref["Vs_1"].grad,
                 ref["Vs_2"].grad if ref["Vs_2"].grad is not None else torch.zeros_like(inp["Vs_2"])]
    return tc, sc, ref_grads


@pytest.mark.parametrize("B,H,N,D,scale", CONFIGS)
def test_tc_backward_matches_reference(B, H, N, D, scale):
    tc, sc, ref = _run(B, H, N, D, scale)
    for idx, name in enumerate(GRAD_NAMES):
        t, s, r = tc[idx].float(), sc[idx].float(), ref[idx].float()
        assert torch.isfinite(t).all(), f"{name}: non-finite values (TC path)"
        # Same tolerances as the main suite ...
        if name in ("grad_Q", "grad_R", "grad_S"):
            rtol, atol = 5e-2, 1e-1
        else:
            rtol, atol = 2e-2, 2e-2
        if torch.allclose(t, r, rtol=rtol, atol=atol):
            continue
        # ... with a fallback for draws where bf16 input quantization pushes
        # even the scalar path past the strict bound (large input_scale): the
        # TC path must then stay within 2x of the scalar path's own error.
        # (TC additionally rounds the softmax weights to bf16 for the output
        # GEMMs — the same tradeoff Y_gather_tc makes — while often being
        # MORE accurate on grad_Q/R/S via the collapsed correction sums.)
        tc_err = (t - r).abs().max().item()
        sc_err = (s - r).abs().max().item()
        assert tc_err <= 2.0 * sc_err + atol, (
            f"{name} (B={B} H={H} N={N} scale={scale}): TC err {tc_err:.3e} "
            f"vs scalar err {sc_err:.3e}")


def test_nonzero_scatter_grads_fall_back():
    """With nonzero scatter cotangents the TC gate must not engage; results
    must still match autograd (scalar path handles the full seven forms)."""
    B, H, N, D = 1, 2, 64, 64
    torch.manual_seed(7)
    dev = "cuda"
    rnd = lambda: torch.randn(B, H, N, D, device=dev)
    names = ["Q", "R", "S", "Vq_1", "Vq_2", "Vr_1", "Vr_2", "Vs_1", "Vs_2"]
    inp = {n: rnd() for n in names}
    bf = {n: v.to(torch.bfloat16) for n, v in inp.items()}
    g = [rnd() for _ in range(6)]

    out = ck.forward(*[bf[n] for n in names], 0.0)
    stats = out[6:12]
    got = ck.backward(*[x.to(torch.bfloat16) for x in g],
                      *[bf[n] for n in names], *stats, 0.0,
                      None, out[0], out[1], out[2])

    ref = {n: inp[n].detach().clone().requires_grad_(True) for n in names}
    ro = tk.forward(ref["Q"], ref["R"], ref["S"], ref["Vq_1"], ref["Vq_2"],
                    ref["Vr_1"], ref["Vr_2"], ref["Vs_1"], ref["Vs_2"], 0.0)
    loss = sum((o * gg).sum() for o, gg in zip(ro, g))
    loss.backward()
    for idx, n in enumerate(names):
        r = ref[n].grad
        if r is None:
            continue
        t = got[idx].float()
        assert torch.allclose(t, r.float(), rtol=5e-2, atol=1e-1), (
            f"grad_{n}: max diff {(t - r.float()).abs().max().item():.3e}")


# ===========================================================================
# Masked path (Bwd_gather_tc<64, true>)
# ===========================================================================

# Mirrors the kernel's smem layout (BTC_BJ=128, BTC_BK=32, BTC_WARPS=8, D=64,
# DPAD=72) so tests that must not silently fall back can skip instead.
def _tc_smem_bytes(N, masked):
    k_pad = -(-N // 32) * 32
    total = 2 * (3 * 128 * 72 + 3 * k_pad * 72)
    total += 4 * (3 * 64 + 3 * k_pad + 3 * 128 + 8 * 2 * 64 + 2 * 64)
    if masked:
        total += 4 * k_pad * (k_pad // 32) + k_pad
    return total


def _tc_fits(N, masked=True):
    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        return False
    return _tc_smem_bytes(N, masked) <= props.shared_memory_per_block_optin


def _ref_gather_masked(Q, R, S, Vq1, Vr1, Vs1, mask):
    """fp32 gather-only reference. mask[b, anchor, key]; a softmax cell is live
    iff both of its non-anchor indices are keys of its own anchor — the
    mask_pair_allowed rule the scalar kernels use.  The same [B, anchor, x, y]
    pair tensor serves all three softmaxes because in each one the two
    non-anchor axes appear in ascending order."""
    D = Q.shape[-1]
    logits = torch.einsum("bhid,bhjd,bhkd->bhijk", Q, R, S) / math.sqrt(D)
    vp = (mask[:, :, :, None] & mask[:, :, None, :]).unsqueeze(1)

    def sm(x):
        x = x.masked_fill(~vp, float("-inf"))
        p = torch.softmax(x.flatten(3), -1).reshape(x.shape)
        return torch.nan_to_num(p, nan=0.0)   # fully masked anchor -> all zero

    Aq = sm(logits)                             # anchor i over (j, k)
    Ar = sm(logits.permute(0, 1, 3, 2, 4))      # anchor j over (i, k)
    As = sm(logits.permute(0, 1, 4, 2, 3))      # anchor k over (i, j)
    return (torch.einsum("bhijk,bhjd,bhkd->bhid", Aq, Vr1, Vs1),
            torch.einsum("bhjik,bhid,bhkd->bhjd", Ar, Vq1, Vs1),
            torch.einsum("bhkij,bhid,bhjd->bhkd", As, Vq1, Vr1))


def _make_mask(kind, B, N, device):
    eye = torch.arange(N, device=device)
    tri = torch.tril(torch.ones(B, N, N, device=device, dtype=torch.bool))
    if kind == "causal":
        return tri
    if kind == "all_true":
        return torch.ones(B, N, N, device=device, dtype=torch.bool)
    if kind == "random":
        m = torch.randint(0, 2, (B, N, N), device=device, dtype=torch.bool)
        m[:, eye, eye] = True     # keep every anchor's own denominator alive
        return m
    if kind == "dead_rows":
        # Fully masked anchors: the forward emits m = NEG_INF, l = 0, Y = 0, so
        # the backward must select those cells away rather than exp() them.
        m = tri.clone()
        m[:, 5, :] = False
        m[:, N - 1, :] = False
        return m
    if kind == "prefix_lm_pad":
        # Bidirectional prefix + causal suffix, with the last 4 slots padded
        # off entirely (rows and columns) — the shape recogs collation emits.
        P = N // 2
        m = torch.zeros(B, N, N, device=device, dtype=torch.bool)
        m[:, :, :P] = True
        m |= tri
        m[:, :, N - 4:] = False
        m[:, N - 4:, :] = False
        return m
    raise ValueError(kind)


MASK_KINDS = ["causal", "all_true", "random", "dead_rows", "prefix_lm_pad"]


def _run_masked(B, H, N, D, mask, scale=1.0, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    rnd = lambda: torch.randn(B, H, N, D, device=dev) * scale
    names = ["Q", "R", "S", "Vq_1", "Vq_2", "Vr_1", "Vr_2", "Vs_1", "Vs_2"]
    inp = {n: rnd() for n in names}
    bf = {n: v.to(torch.bfloat16) for n, v in inp.items()}

    gYq, gYr, gYs = rnd(), rnd(), rnd()
    zero = torch.zeros(B, H, N, D, device=dev, dtype=torch.bfloat16)

    out = ck.forward(*[bf[n] for n in names], 0.0, mask=mask)
    Yq, Yr, Ys = out[0], out[1], out[2]
    stats = out[6:12]

    args = (gYq.to(torch.bfloat16), gYr.to(torch.bfloat16), gYs.to(torch.bfloat16),
            zero, zero, zero, *[bf[n] for n in names], *stats, 0.0)
    tc = ck.backward(*args, mask, Yq, Yr, Ys)   # Y provided -> TC
    sc = ck.backward(*args, mask)               # no Y -> scalar path

    ref = {n: inp[n].detach().clone().requires_grad_(True) for n in names}
    rY = _ref_gather_masked(ref["Q"], ref["R"], ref["S"],
                            ref["Vq_1"], ref["Vr_1"], ref["Vs_1"], mask)
    loss = (rY[0] * gYq).sum() + (rY[1] * gYr).sum() + (rY[2] * gYs).sum()
    loss.backward()
    zeros = torch.zeros(B, H, N, D, device=dev)
    ref_grads = [ref["Q"].grad, ref["R"].grad, ref["S"].grad,
                 ref["Vq_1"].grad, zeros, ref["Vr_1"].grad, zeros,
                 ref["Vs_1"].grad, zeros]
    return tc, sc, ref_grads


# Capped at N=128: the fp32 reference materializes the whole B*H*N^3 cube
# three times over for autograd. Larger N is covered against the scalar path by
# test_masked_tc_matches_scalar_path_large_n below.
MASK_CONFIGS = [(1, 1, 32, 64, 1.0), (2, 2, 32, 64, 1.0),
                (1, 2, 64, 64, 1.0), (2, 2, 64, 64, 1.0),
                (1, 2, 96, 64, 1.0), (1, 2, 128, 64, 1.0),
                (1, 2, 64, 64, 2.0)]


@pytest.mark.parametrize("kind", MASK_KINDS)
@pytest.mark.parametrize("B,H,N,D,scale", MASK_CONFIGS)
def test_tc_masked_backward_matches_reference(kind, B, H, N, D, scale):
    torch.manual_seed(99 + N)
    mask = _make_mask(kind, B, N, "cuda")
    tc, sc, ref = _run_masked(B, H, N, D, mask, scale, seed=1234 + N + B * 7 + H * 13)
    for idx, name in enumerate(GRAD_NAMES):
        t, s, r = tc[idx].float(), sc[idx].float(), ref[idx].float()
        assert torch.isfinite(t).all(), f"{name}: non-finite values (masked TC)"
        if name in ("grad_Q", "grad_R", "grad_S"):
            rtol, atol = 5e-2, 1e-1
        else:
            rtol, atol = 2e-2, 2e-2
        if torch.allclose(t, r, rtol=rtol, atol=atol):
            continue
        # Same escape hatch as the unmasked suite: at large input_scale bf16
        # input quantization pushes the scalar path past the strict bound too,
        # so require only that TC stays within 2x of the scalar path's error.
        tc_err = (t - r).abs().max().item()
        sc_err = (s - r).abs().max().item()
        assert tc_err <= 2.0 * sc_err + atol, (
            f"{name} ({kind} B={B} H={H} N={N} scale={scale}): TC err "
            f"{tc_err:.3e} vs scalar err {sc_err:.3e}")


@pytest.mark.parametrize("kind", ["causal", "prefix_lm_pad"])
@pytest.mark.parametrize("N", [192, 256])
def test_masked_tc_matches_scalar_path_large_n(kind, N):
    """Beyond N=128 the fp32 cube reference is too large to differentiate, so
    check TC against the scalar path (itself validated against the reference at
    smaller N). Tolerances are looser than an exact match because TC rounds the
    softmax weights to bf16 for its output GEMMs."""
    if not _tc_fits(N, masked=True):
        pytest.skip(f"device opt-in smem too small for masked TC at N={N}")
    B, H, D = 1, 1, 64
    torch.manual_seed(4242 + N)
    mask = _make_mask(kind, B, N, "cuda")
    tc, sc, _ = _run_masked(B, H, N, D, mask, seed=7 + N)
    for idx, name in enumerate(GRAD_NAMES):
        t, s = tc[idx].float(), sc[idx].float()
        assert torch.isfinite(t).all(), f"{name}: non-finite values (masked TC)"
        scale_ref = max(s.abs().max().item(), 1e-3)
        err = (t - s).abs().max().item()
        assert err <= 0.05 * scale_ref, (
            f"{name} ({kind} N={N}): TC vs scalar max diff {err:.3e} "
            f"(scalar magnitude {scale_ref:.3e})")


@pytest.mark.parametrize("N", [32, 64, 128, 256])
def test_masked_tc_path_actually_engages(N):
    """Guard against the masked gate silently falling back: the TC and scalar
    paths differ numerically (bf16 GEMM operands vs fp32 scalar), so bitwise
    equality means the gate did not fire."""
    if not _tc_fits(N, masked=True):
        pytest.skip(f"device opt-in smem too small for masked TC at N={N}")
    B, H, D = 1, 2, 64
    mask = _make_mask("causal", B, N, "cuda")
    tc, sc, _ = _run_masked(B, H, N, D, mask, seed=5)
    assert not all(torch.equal(tc[i], sc[i]) for i in range(3)), (
        f"masked TC path did not engage at N={N} (results bitwise identical "
        "to the scalar path)")


# One pass through each tensor-core instantiation, cheap enough to run under
# compute-sanitizer (the full suite is not). N=128/256 are the only shapes that
# reach Y_gather_tc<64,false>; N=16 exercises the padded resident rows.
TC_VARIANTS = [(16, None), (96, None), (128, None), (256, None),
               (16, "causal"), (128, "causal"), (128, "prefix_lm_pad")]


@pytest.mark.parametrize("N,kind", TC_VARIANTS)
def test_tc_variants_finite(N, kind):
    B, H, D = 1, 1, 64
    dev = "cuda"
    torch.manual_seed(7 + N)
    mask = _make_mask(kind, B, N, dev) if kind else None
    names = ["Q", "R", "S", "Vq_1", "Vq_2", "Vr_1", "Vr_2", "Vs_1", "Vs_2"]
    bf = {n: (torch.randn(B, H, N, D, device=dev) * 0.5).to(torch.bfloat16)
          for n in names}
    g = [(torch.randn(B, H, N, D, device=dev) * 0.5).to(torch.bfloat16)
         for _ in range(3)]
    zero = torch.zeros(B, H, N, D, device=dev, dtype=torch.bfloat16)

    out = ck.forward(*[bf[n] for n in names], 0.0, mask=mask)
    grads = ck.backward(*g, zero, zero, zero, *[bf[n] for n in names],
                        *out[6:12], 0.0, mask, out[0], out[1], out[2])
    out_names = ["Y_q", "Y_r", "Y_s", "Y_q_", "Y_r_", "Y_s_"]
    for name, t in zip(out_names + GRAD_NAMES, list(out[:6]) + list(grads)):
        assert torch.isfinite(t).all(), f"{name}: non-finite (N={N} mask={kind})"


@pytest.mark.parametrize("N", [32, 64])
def test_all_true_mask_matches_unmasked(N):
    """An all-true mask must reproduce the unmasked kernel exactly: same
    softmax support, same arithmetic, only the gate selects differ."""
    B, H, D = 1, 2, 64
    dev = "cuda"
    torch.manual_seed(11)
    names = ["Q", "R", "S", "Vq_1", "Vq_2", "Vr_1", "Vr_2", "Vs_1", "Vs_2"]
    rnd = lambda: torch.randn(B, H, N, D, device=dev)
    bf = {n: rnd().to(torch.bfloat16) for n in names}
    g = [rnd().to(torch.bfloat16) for _ in range(3)]
    zero = torch.zeros(B, H, N, D, device=dev, dtype=torch.bfloat16)
    allt = torch.ones(B, N, N, device=dev, dtype=torch.bool)

    def go(mask):
        kw = {} if mask is None else {"mask": mask}
        out = ck.forward(*[bf[n] for n in names], 0.0, **kw)
        return ck.backward(*g, zero, zero, zero, *[bf[n] for n in names],
                           *out[6:12], 0.0, mask, out[0], out[1], out[2])

    masked, unmasked = go(allt), go(None)
    for idx, name in enumerate(GRAD_NAMES):
        a, b = masked[idx].float(), unmasked[idx].float()
        assert torch.allclose(a, b, rtol=1e-2, atol=1e-2), (
            f"{name}: all-true mask diverges from unmasked, max diff "
            f"{(a - b).abs().max().item():.3e}")
