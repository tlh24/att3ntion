"""Correctness oracle for the single query-gather kernels.

The reference is an independent fp64 torch implementation (einsum + one joint
softmax over the flattened (j, k) axis + autograd), fed the same bf16 draws the
kernels see. Thresholds are the handoff's predeclared acceptance targets; the
scale-2.0 rows are stress diagnostics reported separately (-m stress).
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
from att3ntion._single_gather import single_gather_attention

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

NAMES = ["Q", "R", "S", "Vr", "Vs"]
TOL = {  # (rel L2, max err / max(1, ref max-abs))
    "Y": (0.03, 0.05), "dVr": (0.04, 0.05), "dVs": (0.04, 0.05),
    "dQ": (0.06, 0.10), "dR": (0.06, 0.10), "dS": (0.06, 0.10),
}
LSE_MAX = 0.02
MASKS = ["none", "causal", "window16", "random_dead"]


def make_mask(kind, B, N, device, seed=0):
    if kind == "none":
        return None
    i = torch.arange(N, device=device)
    if kind == "causal":
        m = i[None, :] <= i[:, None]
    elif kind == "window16":
        m = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - 16)
    elif kind == "random_dead":
        g = torch.Generator(device="cpu").manual_seed(seed)
        m = torch.rand(B, N, N, generator=g) < 0.6
        m[:, N // 2, :] = False        # one fully masked query per batch
        return m.to(device)
    else:
        raise ValueError(kind)
    return m[None].expand(B, -1, -1).contiguous()


def reference(Q, R, S, Vr, Vs, mask):
    """fp64: Y, LSE and the five input gradients under cotangent dY (set by
    caller via autograd). Never forms a [B,H,N,N,N,D] tensor."""
    D = Q.shape[-1]
    x = torch.einsum("bhid,bhjd,bhkd->bhijk", Q, R, S) / math.sqrt(D)
    if mask is not None:
        vis = (mask[:, :, :, None] & mask[:, :, None, :])[:, None]
        x = x.masked_fill(~vis, float("-inf"))
    flat = x.flatten(3)
    lse = torch.logsumexp(flat, -1)                       # -inf on dead rows
    p = torch.softmax(flat, -1).reshape(x.shape)
    p = torch.nan_to_num(p, nan=0.0)                      # dead rows -> zero
    Y = torch.einsum("bhijk,bhjd->bhikd", p, Vr)
    Y = torch.einsum("bhikd,bhkd->bhid", Y, Vs)
    return Y, lse


def draw(B, H, N, D, seed, scale, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    xs = {n: (torch.randn(B, H, N, D, generator=g) * scale).to(torch.bfloat16).to(device)
          for n in NAMES}
    dY = (torch.randn(B, H, N, D, generator=g) * scale).to(torch.bfloat16).to(device)
    return xs, dY


def run_reference(xs, dY, mask):
    ref = {n: xs[n].double().requires_grad_(True) for n in NAMES}
    Y, lse = reference(*[ref[n] for n in NAMES], mask)
    Y.backward(dY.double())
    grads = {"dQ": ref["Q"].grad, "dR": ref["R"].grad, "dS": ref["S"].grad,
             "dVr": ref["Vr"].grad, "dVs": ref["Vs"].grad}
    return Y.detach(), lse, grads


def errors(actual, ref):
    a, r = actual.double(), ref.double()
    rel = (a - r).norm() / max(r.norm().item(), 1e-12)
    mx = (a - r).abs().max() / max(1.0, r.abs().max().item())
    return rel.item(), mx.item()


def check(name, actual, ref):
    assert torch.isfinite(actual).all(), f"{name}: non-finite"
    if ref.abs().max().item() == 0.0:
        assert actual.abs().max().item() <= 1e-6, f"{name}: nonzero vs identically-zero ref"
        return
    rel, mx = errors(actual, ref)
    rtol, mtol = TOL[name]
    assert rel <= rtol and mx <= mtol, f"{name}: rel L2 {rel:.4f} (<= {rtol}), max {mx:.4f} (<= {mtol})"


def lse_from_ml(m, l):
    return torch.where(l > 0, m.double() + torch.log(l.double().clamp_min(1e-300)),
                       torch.full_like(m.double(), float("-inf")))


def _case(B, H, N, D, seed, scale, kind, api):
    xs, dY = draw(B, H, N, D, seed, scale)
    mask = make_mask(kind, B, N, "cuda", seed)
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)

    if api == "lowlevel":
        assert N % 16 == 0
        Y, m, l = ck.single_gather_forward(*[xs[n] for n in NAMES], mask)
        lse = lse_from_ml(m, l)
        dead = lse_ref == float("-inf")
        assert torch.equal(dead, lse == float("-inf")), "dead-row set differs"
        if (~dead).any():
            lse_err = (lse[~dead] - lse_ref[~dead]).abs().max().item()
            assert lse_err <= LSE_MAX, f"LSE max err {lse_err:.4f}"
        grads = dict(zip(["dQ", "dR", "dS", "dVr", "dVs"],
                         ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y, m, l, mask)))
    else:
        leaves = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
        Y = single_gather_attention(*[leaves[n] for n in NAMES], mask)
        Y.backward(dY)
        grads = {"dQ": leaves["Q"].grad, "dR": leaves["R"].grad, "dS": leaves["S"].grad,
                 "dVr": leaves["Vr"].grad, "dVs": leaves["Vs"].grad}

    check("Y", Y, Y_ref)
    dead = lse_ref == float("-inf")
    if dead.any():
        assert Y[dead].abs().max().item() == 0.0, "dead query rows must produce Y = 0"
    for n in ["dQ", "dR", "dS", "dVr", "dVs"]:
        check(n, grads[n], g_ref[n])


STANDARD = [(1, 2, N, D, seed, scale, kind)
            for D in (64, 128) for N in (16, 33, 65) for seed in (0, 1, 2)
            for scale in (0.5, 1.0) for kind in MASKS]
SMOKE = [(2, 1, 32, 64, 0, 1.0, "causal"), (2, 1, 32, 128, 0, 1.0, "random_dead"),
         (1, 1, 128, 64, 0, 1.0, "none"), (1, 1, 128, 128, 0, 1.0, "window16")]
STRESS = [(1, 2, N, D, seed, 2.0, kind)
          for D in (64, 128) for N in (16, 65) for seed in (0, 1, 2) for kind in MASKS]


@pytest.mark.parametrize("B,H,N,D,seed,scale,kind", STANDARD + SMOKE)
def test_public_bridge(B, H, N, D, seed, scale, kind):
    _case(B, H, N, D, seed, scale, kind, "bridge")


@pytest.mark.parametrize("B,H,N,D,seed,scale,kind",
                         [c for c in STANDARD + SMOKE if c[2] % 16 == 0])
def test_lowlevel(B, H, N, D, seed, scale, kind):
    _case(B, H, N, D, seed, scale, kind, "lowlevel")


@pytest.mark.stress
@pytest.mark.parametrize("B,H,N,D,seed,scale,kind", STRESS)
def test_stress_scale2(B, H, N, D, seed, scale, kind):
    _case(B, H, N, D, seed, scale, kind, "bridge")


def test_reference_finite_difference():
    """The fp64 oracle's own gradient, checked against central differences."""
    torch.manual_seed(0)
    B, H, N, D = 1, 1, 8, 64
    xs = {n: torch.randn(B, H, N, D, dtype=torch.float64, device="cuda") for n in NAMES}
    mask = make_mask("causal", B, N, "cuda")
    dY = torch.randn(B, H, N, D, dtype=torch.float64, device="cuda")

    def loss(**kw):
        return (reference(*[kw[n] for n in NAMES], mask)[0] * dY).sum()

    leaves = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
    loss(**leaves).backward()
    eps = 1e-5
    for n in NAMES:
        for _ in range(3):
            idx = tuple(torch.randint(0, s, (1,)).item() for s in xs[n].shape)
            up, dn = {k: v.clone() for k, v in xs.items()}, {k: v.clone() for k, v in xs.items()}
            up[n][idx] += eps
            dn[n][idx] -= eps
            fd = (loss(**up) - loss(**dn)).item() / (2 * eps)
            assert abs(fd - leaves[n].grad[idx].item()) <= 1e-6 * max(1.0, abs(fd)), n


def test_dead_row_loss_contributes_zero_gradient():
    """A loss supported only on a fully masked query must give zero gradients
    everywhere (that query has no live cells)."""
    B, H, N, D = 1, 1, 32, 64
    xs, _ = draw(B, H, N, D, 3, 1.0)
    mask = make_mask("random_dead", B, N, "cuda", 3)
    dY = torch.zeros(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    dY[:, :, N // 2] = 1.0
    leaves = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
    Y = single_gather_attention(*[leaves[n] for n in NAMES], mask)
    Y.backward(dY)
    assert Y[:, :, N // 2].abs().max().item() == 0.0
    for n in NAMES:
        assert leaves[n].grad.abs().max().item() == 0.0, n


def test_specialized_dispatch_counts():
    xs, dY = draw(1, 1, 64, 64, 0, 1.0)
    before = ck.sg_dispatch()
    Y, m, l = ck.single_gather_forward(*[xs[n] for n in NAMES], None)
    ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y, m, l, None)
    torch.cuda.synchronize()
    after = ck.sg_dispatch()
    assert after["fwd_launches"] - before["fwd_launches"] == 1
    assert after["bwd_anchor_launches"] - before["bwd_anchor_launches"] == 1
    assert after["bwd_rows_launches"] - before["bwd_rows_launches"] == 2
    assert after["last_fwd"].startswith("D=64 ") and "roles=anchor,rows,rows" in after["last_bwd"]


def test_matches_legacy_zero_cotangent_backward():
    """Second diagnostic: the legacy three-gather backward with zero cotangents
    on the unused gathers computes the same five gradients."""
    B, H, N, D = 1, 2, 64, 64
    xs, dY = draw(B, H, N, D, 5, 1.0)
    Y, m, l = ck.single_gather_forward(*[xs[n] for n in NAMES], None)
    got = ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y, m, l, None)
    zero = torch.zeros_like(dY)
    out = ck.forward(xs["Q"], xs["R"], xs["S"], zero, zero, xs["Vr"], zero, xs["Vs"], zero, 0.0)
    leg = ck.backward(dY, zero, zero, zero, zero, zero,
                      xs["Q"], xs["R"], xs["S"], zero, zero, xs["Vr"], zero, xs["Vs"], zero,
                      *out[6:12], 0.0, None, out[0], out[1], out[2])
    for name, a, b in zip(["dQ", "dR", "dS", "dVr", "dVs"], got, [leg[0], leg[1], leg[2], leg[5], leg[7]]):
        rel, mx = errors(a, b)
        assert rel <= 0.02, f"{name} vs legacy: rel L2 {rel:.4f}"
    assert torch.equal(Y, out[0]) and torch.equal(m, out[6]) and torch.equal(l, out[7])


def test_non_default_stream():
    B, H, N, D = 1, 1, 64, 64
    xs, dY = draw(B, H, N, D, 7, 1.0)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        a = {n: xs[n].float().to(torch.bfloat16) * 1 for n in NAMES}   # produced on s
        Y, m, l = ck.single_gather_forward(*[a[n] for n in NAMES], None)
        g = ck.single_gather_backward(dY, *[a[n] for n in NAMES], Y, m, l, None)
        Ys, gs = Y.float() + 0, [x.float() + 0 for x in g]            # consumed on s
    torch.cuda.synchronize()
    Y2, m2, l2 = ck.single_gather_forward(*[xs[n] for n in NAMES], None)
    g2 = ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y2, m2, l2, None)
    assert torch.equal(Ys, Y2.float())
    for x, y in zip(gs, g2):
        assert torch.equal(x, y.float())


@pytest.mark.parametrize("bad", ["dtype", "shape", "D", "N", "mask_shape", "device"])
def test_validation(bad):
    xs, _ = draw(1, 1, 32, 64, 0, 1.0)
    args = [xs[n] for n in NAMES]
    mask = None
    if bad == "dtype":
        args[0] = args[0].float()
    elif bad == "shape":
        args[1] = args[1][:, :, :16].contiguous()
    elif bad == "D":
        args = [x[..., :32].contiguous() for x in args]
    elif bad == "N":
        args = [x[:, :, :24].contiguous() for x in args]
    elif bad == "mask_shape":
        mask = torch.ones(1, 16, 16, dtype=torch.bool, device="cuda")
    elif bad == "device":
        args[2] = args[2].cpu()
    with pytest.raises(RuntimeError):
        ck.single_gather_forward(*args, mask)


# One pass through every kernel instantiation; small enough for compute-sanitizer.
@pytest.mark.sanitizer
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("N,kind", [(16, "none"), (48, "causal"), (128, "none"), (64, "random_dead")])
def test_sanitizer_smoke(D, N, kind):
    xs, dY = draw(1, 1, N, D, 11, 1.0)
    mask = make_mask(kind, 1, N, "cuda", 11)
    Y, m, l = ck.single_gather_forward(*[xs[n] for n in NAMES], mask)
    g = ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y, m, l, mask)
    torch.cuda.synchronize()
    assert torch.isfinite(Y).all() and all(torch.isfinite(x).all() for x in g)
