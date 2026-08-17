"""Correctness tests for the tensor-core paths (Y_gather_tc, Bwd_gather_tc).

The backward TC path engages only when the forward outputs Y_q/Y_r/Y_s are
passed to backward AND the scatter cotangents are zero (scatter unused).
Calling backward WITHOUT the Y tensors forces the scalar path, which lets one
process compare both paths directly, and both against torch autograd.

The MASKED=true variant is covered from `_ref_gather_masked` onward. Since
_torch_kernels has no masked reference, those tests build a gather-only fp32
einsum reference and differentiate it with autograd.

Both gates fail open into the scalar path — wrong D, or too little opt-in shared
memory. Both kernels stream their col side, so both smem costs are flat in N and
both engage at any N. So the dispatch matrix at the end of the file asserts
which kernel ran via ck.tc_launches() rather than trusting the numbers alone.
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
from att3ntion import HypergraphAttention

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
    before = ck.tc_launches()
    got = ck.backward(*[x.to(torch.bfloat16) for x in g],
                      *[bf[n] for n in names], *stats, 0.0,
                      None, out[0], out[1], out[2])
    torch.cuda.synchronize()
    assert ck.tc_launches()[1] == before[1], "TC path engaged despite scatter grads"

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

# Mirror the kernels' smem layouts and dispatch gates. TC_* are forward.cu,
# BTC_* backward.cu; DPAD is D=64 padded against bank conflicts.
TC_BJ, TC_BK, TC_WARPS = 128, 64, 8
BTC_BJ, BTC_BK, BTC_WARPS = 128, 32, 8
DPAD = 64 + 8


def _smem_optin():
    props = torch.cuda.get_device_properties(0)
    return props.shared_memory_per_block_optin if props.major >= 8 else 0


# Independent of N: the forward streams its col side through two TC_BK tiles.
FWD_TC_SMEM = (2 * (2 * TC_BJ * DPAD + 4 * TC_BK * DPAD)
               + 4 * (2 * TC_BK + TC_BJ + TC_WARPS * 64 + TC_WARPS * 2 + 64 + 2 + 64))


# Also independent of N: two staged BTC_BK col tiles + one tile's mask windows.
def _tc_smem_bytes(masked):
    total = 2 * (3 * BTC_BJ * DPAD + 6 * BTC_BK * DPAD)
    total += 4 * (3 * 64 + 6 * BTC_BK + 3 * BTC_BJ + BTC_WARPS * 2 * 64 + 2 * 64)
    if masked:
        total += 4 * 2 * (BTC_BJ + BTC_BK * (BTC_BJ // 32))
        total += 4 * (2 * BTC_BK + BTC_BJ)
    return total


def _tc_fits(masked=True):
    return _tc_smem_bytes(masked) <= _smem_optin()


def _expect_fwd_tc(N, D):
    return D == 64 and FWD_TC_SMEM <= _smem_optin()


def _expect_bwd_tc(N, D, masked):
    return (D == 64 and N % 16 == 0
            and _tc_smem_bytes(masked) <= _smem_optin())


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
    if not _tc_fits(masked=True):
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
    if not _tc_fits(masked=True):
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
    out_names = ["Y_q", "Y_r", "Y_s", "Y_q_", "Y_r_", "Y_s_",
                 "m_i", "l_i", "m_j", "l_j", "m_k", "l_k"]
    for name, t in zip(out_names + GRAD_NAMES, list(out[:12]) + list(grads)):
        assert torch.isfinite(t).all(), f"{name}: non-finite (N={N} mask={kind})"


# ===========================================================================
# Dispatch matrix
# ===========================================================================

def _dispatch_run(B, H, N, D, mask, tc=True, valid=None, seed=0):
    """One forward+backward, reporting TC launches. tc=False forces the scalar
    path in-process so the same inputs can be run down both."""
    torch.manual_seed(seed)
    dev = "cuda"
    names = ["Q", "R", "S", "Vq_1", "Vq_2", "Vr_1", "Vr_2", "Vs_1", "Vs_2"]
    bf = {n: torch.randn(B, H, N, D, device=dev).to(torch.bfloat16) for n in names}
    g = [torch.randn(B, H, N, D, device=dev).to(torch.bfloat16) for _ in range(3)]
    zero = torch.zeros(B, H, N, D, device=dev, dtype=torch.bfloat16)
    v = (N, N, N) if valid is None else (valid, valid, valid)
    if valid is not None:                       # _autograd zero-pads the tail
        for t in list(bf.values()) + g:
            t[:, :, valid:, :] = 0

    prev = ck.tc_set_enabled(tc, tc)
    try:
        before = ck.tc_launches()
        out = ck.forward(*[bf[n] for n in names], 0.0, *v, mask)
        grads = ck.backward(*g, zero, zero, zero, *[bf[n] for n in names],
                            *out[6:12], 0.0, mask, out[0], out[1], out[2])
        torch.cuda.synchronize()
        after = ck.tc_launches()
    finally:
        ck.tc_set_enabled(*prev)
    return bf, out, grads, (after[0] - before[0], after[1] - before[1])


# D: only 64 is TC. N: 128/256 reach the unmasked forward variant, everything
# else the masked one (n_rows % TC_BJ). 384/512 pin that both kernels stream.
DISPATCH_CELLS = [(N, D, kind)
                  for N in [16, 32, 96, 128, 160, 256]
                  for D in [16, 32, 64]
                  for kind in (None, "causal")]
DISPATCH_CELLS += [(272, 64, None), (272, 64, "causal"),
                   (384, 64, None), (512, 64, None), (512, 64, "causal")]


@pytest.mark.parametrize("N,D,kind", DISPATCH_CELLS)
def test_dispatch_selects_expected_path(N, D, kind):
    """A silent fallback is a failure, not a green run. Both directions are
    asserted, so a gate that drifts either way shows up here."""
    B, H = 1, 1
    mask = _make_mask(kind, B, N, "cuda") if kind else None
    _, _, _, (fwd, bwd) = _dispatch_run(B, H, N, D, mask, seed=N + D)
    assert fwd == (3 if _expect_fwd_tc(N, D) else 0), (
        f"forward: {fwd} TC launches at N={N} D={D} mask={kind}")
    assert bwd == (3 if _expect_bwd_tc(N, D, kind is not None) else 0), (
        f"backward: {bwd} TC launches at N={N} D={D} mask={kind}")


@pytest.mark.parametrize("reference", ["scalar", "torch"])
@pytest.mark.parametrize("kind", [None] + MASK_KINDS)
@pytest.mark.parametrize("N", [32, 96, 128, 256, 384, 512])
def test_forward_tc_matches_reference(N, kind, reference):
    """TC-vs-scalar isolates the tensor-core kernel; TC-vs-torch is the actual
    oracle, and catches a wrong assumption both kernels happen to share."""
    B, H, D = 1, 2, 64
    if not _expect_fwd_tc(N, D):
        pytest.skip(f"device opt-in smem too small for forward TC at N={N}")
    if reference == "torch" and N > 128:
        pytest.skip("fp32 cube reference too large past N=128")
    mask = _make_mask(kind, B, N, "cuda") if kind else None
    bf, out, _, (fwd, _) = _dispatch_run(B, H, N, D, mask, seed=N)
    assert fwd == 3, f"forward TC did not engage at N={N} mask={kind}"

    if reference == "scalar":
        _, exp, _, (f2, _) = _dispatch_run(B, H, N, D, mask, tc=False, seed=N)
        assert f2 == 0, "scalar run still took the TC path"
    else:
        f = {n: v.float() for n, v in bf.items()}
        if mask is None:
            exp = tk.forward(f["Q"], f["R"], f["S"], f["Vq_1"], f["Vq_2"],
                             f["Vr_1"], f["Vr_2"], f["Vs_1"], f["Vs_2"], 0.0)
        else:
            exp = _ref_gather_masked(f["Q"], f["R"], f["S"],
                                     f["Vq_1"], f["Vr_1"], f["Vs_1"], mask)

    for idx, name in enumerate(["Y_q", "Y_r", "Y_s"]):
        got, want = out[idx].float(), exp[idx].float()
        assert torch.isfinite(got).all(), f"{name}: non-finite (N={N} {kind})"
        err = (got - want).abs().max().item()
        tol = 0.05 * max(want.abs().max().item(), 1e-3)
        assert err <= tol, (
            f"{name} vs {reference} (N={N} mask={kind}): max diff {err:.3e} "
            f"> {tol:.3e}")


@pytest.mark.parametrize("N,valid", [(32, 19), (128, 100), (256, 200)])
def test_partial_valid_matches_scalar(N, valid):
    """I/J/K_valid < N drives the masked TC variant with no mask tensor — the
    shape _autograd emits for a padded sequence."""
    B, H, D = 1, 2, 64
    if not _expect_fwd_tc(N, D):
        pytest.skip(f"device opt-in smem too small for forward TC at N={N}")
    _, out, grads, (fwd, _) = _dispatch_run(B, H, N, D, None, valid=valid, seed=valid)
    _, exp, ref, (f2, _) = _dispatch_run(B, H, N, D, None, valid=valid,
                                         tc=False, seed=valid)
    assert fwd == 3, f"forward TC did not engage at N={N} valid={valid}"
    assert f2 == 0, "scalar run still took the TC path"
    for idx, name in enumerate(["Y_q", "Y_r", "Y_s"]):
        got, want = out[idx].float()[:, :, :valid], exp[idx].float()[:, :, :valid]
        err = (got - want).abs().max().item()
        assert err <= 0.05 * max(want.abs().max().item(), 1e-3), (
            f"{name} (N={N} valid={valid}): TC vs scalar max diff {err:.3e}")
    for idx, name in enumerate(GRAD_NAMES[:3]):
        got, want = grads[idx].float()[:, :, :valid], ref[idx].float()[:, :, :valid]
        err = (got - want).abs().max().item()
        assert err <= 0.05 * max(want.abs().max().item(), 1e-3), (
            f"{name} (N={N} valid={valid}): TC vs scalar max diff {err:.3e}")


# _autograd pads to a multiple of 16 while TC_BJ is 128, so real (ragged)
# sequence lengths reach the kernels at N values no fixed sweep generates.
@pytest.mark.parametrize("N", [16, 19, 32, 48, 100, 129])
@pytest.mark.parametrize("kind", [None, "causal"])
def test_autograd_padded_n_engages_tc(N, kind):
    torch.manual_seed(N)
    B, d_model, H = 2, 128, 2          # d_head = 64, the one TC shape
    mod = HypergraphAttention(d_model=d_model, n_heads=H, dropout_rate=0.0,
                              scatter=False).to("cuda", torch.float32)
    x = torch.randn(B, N, d_model, device="cuda")
    mask = _make_mask(kind, B, N, "cuda") if kind else None

    before = ck.tc_launches()
    y = mod(x, mask=mask)
    y.sum().backward()
    torch.cuda.synchronize()
    fwd, bwd = (a - b for a, b in zip(ck.tc_launches(), before))

    padded = -(-N // 16) * 16
    assert torch.isfinite(y).all()
    assert fwd == (3 if _expect_fwd_tc(padded, 64) else 0), (
        f"forward: {fwd} TC launches at N={N} (padded {padded}) mask={kind}")
    assert bwd == (3 if _expect_bwd_tc(padded, 64, kind is not None) else 0), (
        f"backward: {bwd} TC launches at N={N} (padded {padded}) mask={kind}")


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
