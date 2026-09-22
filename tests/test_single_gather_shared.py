"""Oracle tests for the shared-KV prototype (docs/KERNEL_HISTORY.md).
The fp64 reference takes the shared tensors as leaves and expands them
over query heads, so its KV gradients are the sums over heads the kernels must return."""
import math
import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import att3ntion._cuda_kernels as ck
from att3ntion._single_gather import window_mask
from att3ntion._single_gather_shared import single_gather_shared_attention
from test_single_gather import TOL, LSE_MAX, check, errors, lse_from_ml, reference

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

NAMES = ["Q", "R", "S", "Vr", "Vs"]
GRADS = ["dQ", "dR", "dS", "dVr", "dVs"]
D, W = 128, 32
GROUPS = [1, 2, 4]


def available(g):
    """Grouped kernels land in stage B; skip (never pass) while the launcher says unavailable."""
    if g == 1:
        return
    xs, dY = draw(1, 8, 32, 7)
    mask = full_mask(1, 32)
    try:
        ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, W, g)
        Y, m, l = ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, W, 1)
        ck.single_gather_shared_backward(dY, *[xs[n] for n in NAMES], Y, m, l, mask, W, g)
    except RuntimeError as e:
        if "unavailable" in str(e):
            pytest.skip(f"grouped kernels (G={g}) not built: {e}")
        raise


def draw(B, Hq, N, seed, scale=1.0, qrs_scale=None, dy_scale=1.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s_qrs = qrs_scale if qrs_scale is not None else scale
    xs = {"Q": (torch.randn(B, Hq, N, D, generator=g) * s_qrs).to(torch.bfloat16).to(device)}
    for n in ("R", "S"):
        xs[n] = (torch.randn(B, 1, N, D, generator=g) * s_qrs).to(torch.bfloat16).to(device)
    for n in ("Vr", "Vs"):
        xs[n] = (torch.randn(B, 1, N, D, generator=g) * scale).to(torch.bfloat16).to(device)
    dY = (torch.randn(B, Hq, N, D, generator=g) * dy_scale).to(torch.bfloat16).to(device)
    return xs, dY


def full_mask(B, N, device="cuda"):
    return window_mask(N, W, device)[None].expand(B, -1, -1).contiguous()


def run_reference(xs, dY, mask):
    """Shared KV as leaves, expanded over heads inside the graph."""
    Hq = xs["Q"].size(1)
    leaves = {n: xs[n].double().requires_grad_(True) for n in NAMES}
    exp = {n: (leaves[n] if n == "Q" else leaves[n].expand(-1, Hq, -1, -1)) for n in NAMES}
    Y, lse = reference(*[exp[n] for n in NAMES], mask)
    Y.backward(dY.double())
    return Y.detach(), lse, {g: leaves[n].grad for g, n in zip(GRADS, NAMES)}


def bridge(xs, dY, mask, fg, rg):
    leaves = {n: xs[n].clone().requires_grad_(True) for n in NAMES}
    Y = single_gather_shared_attention(*[leaves[n] for n in NAMES], mask, fwd_group=fg, rs_group=rg, window=W)
    Y.backward(dY)
    return Y, {g: leaves[n].grad for g, n in zip(GRADS, NAMES)}


def check_all(Y, grads, Y_ref, lse_ref, g_ref):
    check("Y", Y, Y_ref)
    dead = lse_ref == float("-inf")
    if dead.any():
        assert Y[dead].abs().max().item() == 0.0, "dead query rows must produce Y = 0"
    for g in GRADS:
        check(g, grads[g], g_ref[g])
    for g in GRADS[1:]:
        assert tuple(grads[g].shape) == tuple(g_ref[g].shape) and grads[g].size(1) == 1, "KV gradients must stay [B,1,N,D]"


@pytest.mark.parametrize("G", GROUPS)
@pytest.mark.parametrize("N", [31, 32, 33, 65])
@pytest.mark.parametrize("seed", [7, 11])
def test_bridge_shapes(G, N, seed):
    available(G)
    xs, dY = draw(1, 8, N, seed)
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, None if False else full_mask(1, N))
    Y, grads = bridge(xs, dY, None, G, G)
    check_all(Y, grads, Y_ref, lse_ref, g_ref)


@pytest.mark.parametrize("G", GROUPS)
def test_batch_and_head_strides(G):
    available(G)
    xs, dY = draw(2, 8, 33, 7)
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, full_mask(2, 33))
    Y, grads = bridge(xs, dY, None, G, G)
    check_all(Y, grads, Y_ref, lse_ref, g_ref)


@pytest.mark.sanitizer
@pytest.mark.parametrize("G", GROUPS)
def test_dead_query_and_removed_keys(G):
    """One fully masked query and a few visible keys removed inside the window."""
    available(G)
    B, Hq, N = 1, 8, 64
    xs, dY = draw(B, Hq, N, 3)
    extra = torch.ones(B, N, N, dtype=torch.bool, device="cuda")
    extra[:, 40, :] = False                       # dead query
    extra[:, 50, 45] = False                      # removed inside the window
    extra[:, 33, 20:24] = False
    mask = full_mask(B, N) & extra
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)
    Y, grads = bridge(xs, dY, extra, G, G)
    check_all(Y, grads, Y_ref, lse_ref, g_ref)
    Yl, m, l = ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, W, G)
    lse = lse_from_ml(m, l)
    dead = lse_ref == float("-inf")
    assert torch.equal(dead, lse == float("-inf"))
    assert (lse[~dead] - lse_ref[~dead]).abs().max().item() <= LSE_MAX


@pytest.mark.parametrize("G", GROUPS)
def test_single_head_cotangent_isolates_heads(G):
    """dY nonzero for one query head only: other dQ heads are exactly zero and the
    shared KV gradients equal that head's contribution alone."""
    available(G)
    B, Hq, N = 1, 8, 48
    xs, dY = draw(B, Hq, N, 5)
    dY = torch.zeros_like(dY)
    dY[:, 3] = torch.randn(B, N, D, device="cuda").to(torch.bfloat16)
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, full_mask(B, N))
    Y, grads = bridge(xs, dY, None, G, G)
    check_all(Y, grads, Y_ref, lse_ref, g_ref)
    others = [h for h in range(Hq) if h != 3]
    assert grads["dQ"][:, others].abs().max().item() == 0.0


@pytest.mark.parametrize("G", GROUPS)
def test_repeated_backward_has_no_stale_state(G):
    available(G)
    B, Hq, N = 1, 8, 64
    xs, dY = draw(B, Hq, N, 9)
    mask = full_mask(B, N)
    Y, m, l = ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, W, G)
    g1 = ck.single_gather_shared_backward(dY, *[xs[n] for n in NAMES], Y, m, l, mask, W, G)
    g2 = ck.single_gather_shared_backward(dY, *[xs[n] for n in NAMES], Y, m, l, mask, W, G)
    for a, b in zip(g1, g2):
        assert torch.equal(a, b), "second call on the same state must reproduce the first"
    xs2, dY2 = draw(B, Hq, N, 10)
    Y2, m2, l2 = ck.single_gather_shared_forward(*[xs2[n] for n in NAMES], mask, W, G)
    g3 = ck.single_gather_shared_backward(dY2, *[xs2[n] for n in NAMES], Y2, m2, l2, mask, W, G)
    Y_ref, lse_ref, g_ref = run_reference(xs2, dY2, mask)
    for g, t in zip(GRADS, g3):
        check(g, t, g_ref[g])


def test_rejections():
    xs, dY = draw(1, 8, 32, 0)
    mask = full_mask(1, 32)
    with pytest.raises(RuntimeError):      # unsupported window (16 is supported)
        ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, 24, 1)
    with pytest.raises(RuntimeError):      # Hkv != 1
        ck.single_gather_shared_forward(xs["Q"], xs["R"].expand(-1, 8, -1, -1).contiguous(), xs["S"], xs["Vr"], xs["Vs"], mask, W, 1)
    with pytest.raises(RuntimeError):      # D
        ck.single_gather_shared_forward(*[x[..., :64].contiguous() for x in (xs[n] for n in NAMES)], mask, W, 1)
    with pytest.raises(RuntimeError):      # Hq not divisible by the group
        xs3, _ = draw(1, 6, 32, 0)
        ck.single_gather_shared_forward(*[xs3[n] for n in NAMES], mask, W, 4)
    with pytest.raises((RuntimeError, TypeError)):      # mask required
        ck.single_gather_shared_forward(*[xs[n] for n in NAMES], None, W, 1)


def test_native_matches_replicated_path_bitwise():
    """With matching tight tiles, native and replicated-KV forward/dQ agree
    bitwise, and the reduced KV gradients retain the adapter's cast points."""
    B, Hq, N = 2, 8, 64
    xs, dY = draw(B, Hq, N, 21)
    mask = full_mask(B, N)
    Y, m, l = ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, W, 1)
    rep = {n: (xs[n] if n == "Q" else xs[n].expand(-1, Hq, -1, -1).contiguous()) for n in NAMES}
    ck.sg_set_config({k: 0 for k in ck.sg_get_config()})
    ck.sg_set_config({"fwd_warps": 2, "fwd_bk": 16, "bwd_a_warps": 2, "bwd_a_bk": 32, "bwd_r_warps": 2, "bwd_r_bk": 16, "bwd_dh": 128})
    try:
        Yr, mr, lr = ck.single_gather_forward(*[rep[n] for n in NAMES], mask, W)
        gr = ck.single_gather_backward(dY, *[rep[n] for n in NAMES], Yr, mr, lr, mask, W)
    finally:
        ck.sg_set_config({k: 0 for k in ck.sg_get_config()})
    assert torch.equal(Y, Yr) and torch.equal(m, mr) and torch.equal(l, lr)
    g = ck.single_gather_shared_backward(dY, *[xs[n] for n in NAMES], Y, m, l, mask, W, 1)
    assert "Q=Bwd_gather_tc<128,masked,2,32,anchor,dh128>" in ck.sg_dispatch()["last_shared_bwd"]
    assert torch.equal(g[0], gr[0])
    for a, b in zip(g[1:], gr[1:]):
        want = b.float().sum(1, keepdim=True).to(torch.bfloat16)
        rel, mx = errors(a, want)
        assert rel <= 1e-3, f"reduction differs from the adapter contract: rel L2 {rel}"


@pytest.mark.stress
@pytest.mark.parametrize("G", GROUPS)
@pytest.mark.parametrize("case", [("scale2", 0), ("scale2", 1), ("concentrated", 0)])
def test_stress_shared(G, case):
    available(G)
    kind, seed = case
    if kind == "scale2":
        xs, dY = draw(1, 8, 32, seed, scale=2.0)
    else:
        xs, dY = draw(1, 8, 32, seed, scale=1.0, qrs_scale=3.0)
    mask = full_mask(1, 32)
    Y_ref, lse_ref, g_ref = run_reference(xs, dY, mask)
    Y, grads = bridge(xs, dY, None, G, G)
    check_all(Y, grads, Y_ref, lse_ref, g_ref)


@pytest.mark.parametrize("G", [2, 4])
def test_grouped_rounding_contract(G):
    """G2 preserves row summation order. The optimized G4 strategy uses BJ16;
    it must pass the same FP64 tolerances, but need not match BJ32 bitwise.
    Forward and dQ arithmetic remains identical for both strategies."""
    available(G)
    B, Hq, N = 2, 8, 96
    xs, dY = draw(B, Hq, N, 31)
    mask = full_mask(B, N)
    Yn, mn, ln = ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, W, 1)
    gn = ck.single_gather_shared_backward(dY, *[xs[n] for n in NAMES], Yn, mn, ln, mask, W, 1)
    Yg, mg, lg = ck.single_gather_shared_forward(*[xs[n] for n in NAMES], mask, W, G)
    gg = ck.single_gather_shared_backward(dY, *[xs[n] for n in NAMES], Yg, mg, lg, mask, W, G)
    torch.cuda.synchronize()
    assert torch.equal(Yn, Yg) and torch.equal(mn, mg) and torch.equal(ln, lg)
    assert torch.equal(gn[0], gg[0])
    if G == 2:
        for a, b in zip(gn, gg):
            assert torch.equal(a, b)
    else:
        _, _, reference_grads = run_reference(xs, dY, mask)
        for name, grad in zip(GRADS, gg):
            check(name, grad, reference_grads[name])
    assert f"fwd_group={G}" in ck.sg_dispatch()["last_shared_fwd"] and f"rs_group={G}" in ck.sg_dispatch()["last_shared_bwd"]
