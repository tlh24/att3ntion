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
    # Per (b, h) as well, so one broken head cannot hide inside the aggregate.
    for b in range(actual.size(0)):
        for h in range(actual.size(1)):
            if ref[b, h].abs().max().item() == 0.0:
                continue
            rel, mx = errors(actual[b, h], ref[b, h])
            assert rel <= rtol and mx <= mtol, f"{name}[b={b},h={h}]: rel L2 {rel:.4f}, max {mx:.4f}"


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


# ---------------------------------------------------------------------------
# Structured tile bounds (window metadata), boundary lengths, tile shapes
# ---------------------------------------------------------------------------

@pytest.fixture
def default_tiles():
    yield
    ck.sg_set_config({k: 0 for k in ck.sg_get_config()})


def _window(N, w):
    i = torch.arange(N)
    return (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - w)


@pytest.mark.parametrize("N,w", [(16, 16), (48, 48), (64, 7), (64, 16), (96, 32), (128, 128), (128, 1), (256, 32), (272, 40)])
@pytest.mark.parametrize("bj,bk", [(32, 16), (64, 32), (128, 32), (128, 64)])
def test_tile_ranges_cover_visible_pairs_exactly_once(N, w, bj, bk):
    """Host enumerator over the bounds the kernels run: for every CTA (anchor a,
    both query sides) every visible (row, col) pair falls in exactly one
    visited tile, and the dense setting (window 0) visits every tile."""
    mask = _window(N, w)
    # A tail tile is padded relative to its shifted start, so it can extend
    # beyond ceil(N / tile) * tile. Reserve a full tile beyond N on each axis.
    Kp = N + bk
    dense = -(-N // bj) * -(-N // bk)
    visited = 0
    for side in (0, 1):
        for a in range(N):
            ranges = ck.sg_tile_ranges(N, w, a, side, bj, bk)
            cover = torch.zeros(N + bj, Kp, dtype=torch.int32)
            last = -1
            for j0, k_lo, k_hi in ranges:
                assert 0 <= j0 < N and 0 <= k_lo < N
                assert (k_hi - k_lo) % bk == 0 and k_lo < k_hi < Kp
                assert j0 > last, "row blocks must be visited once, in order"
                last = j0
                cover[j0:j0 + bj, k_lo:k_hi] += 1
                visited += (k_hi - k_lo) // bk
            assert cover.max().item() <= 1
            if side == 0:       # query is the anchor: rows j and cols k both visible to a
                need = mask[a][:, None] & mask[a][None, :]
            else:               # anchor is a key: queries i that see it, and the cols they see
                need = mask[:, a][:, None] & mask
            assert bool(cover[:N, :N][need].all()), f"side {side} anchor {a}: visible pair skipped"
            full = ck.sg_tile_ranges(N, 0, a, side, bj, bk)
            assert sum((k_hi - k_lo) // bk for _, k_lo, k_hi in full) == dense
    assert visited <= 2 * N * dense
    if 2 * w <= N and 2 * bj <= N and 2 * bk <= N:
        assert visited < 2 * N * dense, "a window half the sequence must skip tiles"


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.sanitizer
@pytest.mark.parametrize("N,w", [(64, 64), (80, 7), (80, 31), (80, 33), (96, 40),
                                 (128, 16), (256, 256), (256, 32), (48, 48)])
def test_shifted_window_metadata_matches_oracle(D, N, w):
    """Tight starts change BF16 grouping. Both dense and bounded traversals
    must meet the original oracle tolerances, including holes and a dead row."""
    xs, dY = draw(1, 2, N, D, 4, 1.0)
    mask = _window(N, w)[None].expand(1, -1, -1).contiguous().cuda()
    mask[:, N // 2, :] = False
    mask[:, -1, max(0, N - w):N:3] = False
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)
    Y0, m0, l0 = ck.single_gather_forward(*[xs[n] for n in NAMES], mask)
    g0 = ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y0, m0, l0, mask)
    Y1, m1, l1 = ck.single_gather_forward(*[xs[n] for n in NAMES], mask, w)
    g1 = ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y1, m1, l1, mask, w)
    torch.cuda.synchronize()
    dead = lse_ref == float("-inf")
    for Y, m, l, grads in [(Y0, m0, l0, g0), (Y1, m1, l1, g1)]:
        check("Y", Y, Y_ref)
        assert Y[dead].abs().max().item() == 0.0
        lse = lse_from_ml(m, l)
        assert torch.equal(dead, lse == float("-inf"))
        assert (lse[~dead] - lse_ref[~dead]).abs().max().item() <= LSE_MAX
        for name, grad in zip(["dQ", "dR", "dS", "dVr", "dVs"], grads):
            check(name, grad, g_ref[name])
    assert "win=%d" % w in ck.sg_dispatch()["last_bwd"]


def test_window_metadata_requires_mask():
    xs, _ = draw(1, 1, 32, 64, 0, 1.0)
    with pytest.raises(RuntimeError):
        ck.single_gather_forward(*[xs[n] for n in NAMES], None, 16)


@pytest.mark.parametrize("N", [33, 65, 100])
def test_bridge_builds_window_mask(N):
    """window= without a mask builds the equal causal window (padding stays
    invisible) and matches the explicit-mask call and the oracle."""
    B, H, D, w = 1, 2, 64, 16
    xs, dY = draw(B, H, N, D, 9, 1.0)
    mask = _window(N, w)[None].expand(B, -1, -1).contiguous().cuda()
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)
    a = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
    Ya = single_gather_attention(*[a[n] for n in NAMES], window=w)
    Ya.backward(dY)
    b = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
    Yb = single_gather_attention(*[b[n] for n in NAMES], mask, window=w)
    Yb.backward(dY)
    assert torch.equal(Ya, Yb)
    for n in NAMES:
        assert torch.equal(a[n].grad, b[n].grad)
    check("Y", Ya, Y_ref)
    for g, n in zip(["dQ", "dR", "dS", "dVr", "dVs"], NAMES):
        check(g, a[n].grad, g_ref[g])


# Boundary lengths around every tile size, a covering set rather than the
# Cartesian product: D, mask kind and batch/head shape rotate through the list;
# the window cases also run with the metadata declared.
BOUNDARY_N = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
_KINDS = ["causal", "window16", "none", "random_dead"]
_SHAPES = [(1, 2), (2, 1), (1, 3)]
BOUNDARY = [(N, 64 if i % 2 == 0 else 128, _KINDS[i % 4], *_SHAPES[i % 3]) for i, N in enumerate(BOUNDARY_N)]


@pytest.mark.parametrize("N,D,kind,B,H", BOUNDARY)
def test_boundary_lengths(N, D, kind, B, H):
    xs, dY = draw(B, H, N, D, 13, 1.0)
    mask = make_mask(kind, B, N, "cuda", 13)
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)
    window = {"causal": N, "window16": 16}.get(kind, 0)
    for win in ({0, window} if window else {0}):
        leaves = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
        Y = single_gather_attention(*[leaves[n] for n in NAMES], mask, window=win)
        Y.backward(dY)
        check("Y", Y, Y_ref)
        dead = lse_ref == float("-inf")
        if dead.any():
            assert Y[dead].abs().max().item() == 0.0
        for g, n in zip(["dQ", "dR", "dS", "dVr", "dVs"], NAMES):
            check(g, leaves[n].grad, g_ref[g])


TILE_CONFIGS = [dict(fwd_warps=2, fwd_bk=16), dict(fwd_warps=4, fwd_bk=64), dict(fwd_warps=8, fwd_bk=16),
                dict(fwd_warps=8, fwd_bk=32), dict(bwd_a_warps=2, bwd_a_bk=16, bwd_r_warps=2, bwd_r_bk=16),
                dict(bwd_a_warps=4, bwd_a_bk=32, bwd_r_warps=4, bwd_r_bk=16), dict(bwd_a_warps=8, bwd_a_bk=16, bwd_r_warps=8, bwd_r_bk=16),
                dict(bwd_dh=128), dict(bwd_dh=128, bwd_a_warps=4, bwd_a_bk=16, bwd_r_warps=4, bwd_r_bk=32)]


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("cfg", TILE_CONFIGS)
@pytest.mark.parametrize("kind", ["causal", "random_dead"])
def test_tile_shape_variants(default_tiles, D, cfg, kind):
    """Every tile shape the tuner may pick passes the oracle (the D=128
    one-pass output variant is illegal at D=64 and must raise)."""
    if cfg.get("bwd_dh") == 128 and D == 64:
        ck.sg_set_config(cfg)
        xs, dY = draw(1, 1, 32, D, 0, 1.0)
        Y, m, l = ck.single_gather_forward(*[xs[n] for n in NAMES], None)
        with pytest.raises(RuntimeError):
            ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y, m, l, None)
        return
    ck.sg_set_config(cfg)
    B, H, N = 1, 2, 80
    xs, dY = draw(B, H, N, D, 21, 1.0)
    mask = make_mask(kind, B, N, "cuda", 21)
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)
    win = N if kind == "causal" else 0
    Y, m, l = ck.single_gather_forward(*[xs[n] for n in NAMES], mask, win)
    grads = dict(zip(["dQ", "dR", "dS", "dVr", "dVs"],
                     ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y, m, l, mask, win)))
    disp = ck.sg_dispatch()
    for k, v in cfg.items():
        if k.startswith("fwd"):
            assert f"{k[4:]}={v}" in disp["last_fwd"]
    check("Y", Y, Y_ref)
    lse = lse_from_ml(m, l)
    dead = lse_ref == float("-inf")
    if (~dead).any():
        assert (lse[~dead] - lse_ref[~dead]).abs().max().item() <= LSE_MAX
    for g in grads:
        check(g, grads[g], g_ref[g])


def test_illegal_tile_shape_raises(default_tiles):
    xs, dY = draw(1, 1, 32, 64, 0, 1.0)
    ck.sg_set_config(dict(fwd_bk=48))
    with pytest.raises(RuntimeError):
        ck.single_gather_forward(*[xs[n] for n in NAMES], None)
    ck.sg_set_config(dict(fwd_bk=0, bwd_r_bk=64))
    Y, m, l = ck.single_gather_forward(*[xs[n] for n in NAMES], None)
    with pytest.raises(RuntimeError):
        ck.single_gather_backward(dY, *[xs[n] for n in NAMES], Y, m, l, None)
    with pytest.raises(ValueError):
        ck.sg_set_config(dict(nonsense=1))


# Numerical diagnostics beyond the std-2.0 stress rows: scale sweeps, extreme
# upstream gradients, concentrated (peaky) and near-uniform softmaxes. Same
# thresholds, reported separately (-m stress); frozen before final selection.
def draw_dist(B, H, N, D, seed, dist, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s_qrs, s_v, s_dy = {"scale0.25": (0.25, 0.25, 1.0), "scale4": (4.0, 4.0, 1.0), "dy1e-3": (1.0, 1.0, 1e-3),
                        "dy1e3": (1.0, 1.0, 1e3), "concentrated": (3.0, 1.0, 1.0), "uniform": (0.05, 1.0, 1.0),
                        "cancel": (1.0, 1.0, 1.0)}[dist]
    xs = {}
    for n in NAMES:
        sc = s_qrs if n in ("Q", "R", "S") else s_v
        xs[n] = (torch.randn(B, H, N, D, generator=g) * sc).to(torch.bfloat16).to(device)
    if dist == "cancel":        # values with a large common offset: Y is a small difference of big numbers
        xs["Vr"] = (xs["Vr"].float() + 8.0).to(torch.bfloat16)
        xs["Vs"] = (xs["Vs"].float() - 8.0).to(torch.bfloat16)
    dY = (torch.randn(B, H, N, D, generator=g) * s_dy).to(torch.bfloat16).to(device)
    return xs, dY


DISTS = ["scale0.25", "scale4", "dy1e-3", "dy1e3", "concentrated", "uniform", "cancel"]


@pytest.mark.stress
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("seed", [0, 1])
def test_stress_distributions(D, dist, seed):
    B, H, N = 1, 2, 65
    xs, dY = draw_dist(B, H, N, D, seed, dist)
    mask = make_mask("causal", B, N, "cuda")
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)
    leaves = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
    Y = single_gather_attention(*[leaves[n] for n in NAMES], mask, window=N)
    Y.backward(dY)
    check("Y", Y, Y_ref)
    for g, n in zip(["dQ", "dR", "dS", "dVr", "dVs"], NAMES):
        check(g, leaves[n].grad, g_ref[g])
