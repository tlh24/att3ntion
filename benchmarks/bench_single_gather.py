"""Single query-gather comparison: att3ntion CUDA vs the Fast and Simplex paper
kernels and FBGEMM's Triton/TLX providers.

    python benchmarks/bench_single_gather.py --suite correctness
    python benchmarks/bench_single_gather.py --suite core --sessions 5 --rounds 7
    python benchmarks/bench_single_gather.py --suite bh,batch,sharing,mask,seq
    python benchmarks/bench_single_gather.py --suite tune --candidates <grid.json>
    python benchmarks/bench_single_gather.py --analyze <run_dir> [<run_dir> ...]
    python benchmarks/bench_single_gather.py --rerender <old_run_dir>

Every provider computes the same operator on the same seeded bf16 inputs:
one softmax per query over (j, k) pairs with causal windows w on both key
axes (w = N is full causal; the `unmasked` regime has no mask), scale
1/sqrt(D), no biases. Correctness is gated against a full-shape fp64 oracle
(dense for small shapes, streamed over head/query chunks for large ones)
before anything is timed. Gates are per operation: a forward needs Y and, when
the provider claims a natural-log LSE, that statistic; a backward needs its own
forward state validated plus all five gradients; combined and API operations
need both. A ratio is published only when both operands pass the gate of the
operation being compared. Event ratios come from CUDA-event medians and wall
ratios from synchronized wall-clock medians, in separately named columns.

Device latency is CUDA events around the warmed call with output allocation
inside; the backward includes the delta reduction, bf16 conversions and, for
shared KV, the reduction of the replicated KV gradients to the logical KV
shape (`bwd_prepared` keeps the old reduction-free component metric). API
latency is synchronized wall clock around the call including layout
conversion, KV replication and gradient reduction to the logical shapes.

Sessions are independent processes with their own provider-order seed; the
analysis pairs provider and reference within (session, round) and bootstraps
that hierarchy for the 95% interval, so inner-loop iterations are never
treated as independent replicates.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

try:
    import att3ntion._cuda_kernels as ck  # noqa: E402
except Exception as _e:      # noqa: BLE001  (analysis/rerender/tests need no GPU build)
    ck = None
    _CK_ERR = repr(_e)

NAMES = ["Q", "R", "S", "Vr", "Vs"]
GRADS = ["dQ", "dR", "dS", "dVr", "dVs"]
TOL = {"Y": (0.03, 0.05), "dVr": (0.04, 0.05), "dVs": (0.04, 0.05),
       "dQ": (0.06, 0.10), "dR": (0.06, 0.10), "dS": (0.06, 0.10)}
LSE_MAX = 0.02
REPEAT_MAX_REL_L2 = 0.01           # two backward calls on the same state must agree to this
ORACLE_DENSE_BUDGET = 2 ** 27      # fp64 cube elements the dense reference may hold
ORACLE_CHUNK_BUDGET = 2 ** 25      # cube elements per streamed chunk
V0_EXT_DEFAULT = "/home/dev/v0_cf1e258/att3ntion/_cuda_kernels.cpython-311-x86_64-linux-gnu.so"
TIMED_OPS = ("fwd", "bwd", "bwd_prepared", "fwd_bwd", "legacy_state_fwd", "api_fwd", "api_fwd_bwd")


# ---------------------------------------------------------------------------
# Configurations and families
# ---------------------------------------------------------------------------

def make_cfg(B, Hq, Hkv, N, D, w=None, mask="window", kv_repr="shared", families=()):
    """One matrix cell. mask="window": causal window w on both key axes (w=N
    or None is full causal); mask="none": no mask at all (every pair visible).
    kv_repr="repeated": identical KV values physically repeated across the Hq
    heads and treated as independent heads (per-head gradients), the
    representation control for the shared-KV cells."""
    w = N if (w is None or w >= N) else w
    c = dict(B=B, Hq=Hq, Hkv=Hkv, N=N, D=D, w=w, mask=mask, kv_repr=kv_repr)
    c["families"] = list(families)
    c["name"] = cfg_name(c)
    return c


def cfg_name(c):
    if c["mask"] == "none":
        tail = "unmasked"
    else:
        tail = "wfull" if c["w"] == c["N"] else f"w{c['w']}"
    heads = f"H{c['Hq']}" if c["Hkv"] == c["Hq"] else f"Hq{c['Hq']}_Hkv{c['Hkv']}"
    if c.get("kv_repr") == "repeated":
        heads += "rep"
    return f"B{c['B']}_{heads}_D{c['D']}_N{c['N']}_{tail}"


def families():
    F = {}
    F["core"] = [make_cfg(1, H, H, N, D, w, families=["core"])
                 for H in (1, 4) for D in (64, 128) for N in (128, 256) for w in (N, 32)]
    F["shared-kv"] = [make_cfg(1, 64, 1, N, 128, w, families=["shared-kv"])
                      for N in (128, 256, 512) for w in (N, 32)]
    F["bh"] = [make_cfg(B, H, H, 256, 128, families=["bh"])
               for B, H in [(1, 1), (1, 4), (1, 16), (1, 64), (4, 1), (4, 4), (4, 16), (16, 1), (16, 4), (64, 1)]]
    F["batch"] = [make_cfg(B, H, H, 256, D, families=["batch"])
                  for H in (1, 4) for D in (64, 128) for B in (1, 4, 16)]
    F["sharing"] = ([make_cfg(1, 64, Hkv, 256, 128, families=["sharing"]) for Hkv in (64, 8, 1)]
                    + [make_cfg(1, 64, Hkv, N, 128, families=["sharing"]) for N in (128, 512) for Hkv in (64, 1)]
                    + [make_cfg(1, 64, 64, 256, 128, kv_repr="repeated", families=["sharing"])])
    F["mask"] = [make_cfg(B, 4, 4, 256, D, w, mask=m, families=["mask"])
                 for B, D in [(1, 128), (4, 64)]
                 for m, w in [("none", None), ("window", None), ("window", 32), ("window", 64), ("window", 128)]]
    F["seq"] = [make_cfg(1, H, H, N, 128, families=["seq"]) for H in (1, 4, 64) for N in (128, 256, 512)]
    F["anchors"] = [make_cfg(1, 1, 1, 128, 64, families=["anchors"]),
                    make_cfg(1, 4, 4, 256, 128, families=["anchors"]),
                    make_cfg(4, 4, 4, 256, 64, families=["anchors"]),
                    make_cfg(16, 4, 4, 256, 128, families=["anchors"]),
                    make_cfg(1, 64, 64, 256, 128, families=["anchors"]),
                    make_cfg(1, 64, 1, 256, 128, families=["anchors"]),
                    make_cfg(1, 4, 4, 256, 128, 32, families=["anchors"]),
                    make_cfg(1, 64, 1, 256, 128, 32, families=["anchors"]),
                    make_cfg(1, 4, 4, 256, 128, mask="none", families=["anchors-unmasked"])]
    F["holdout"] = [make_cfg(2, 2, 2, 192, 64, families=["holdout"]),
                    make_cfg(8, 8, 8, 384, 128, families=["holdout"]),
                    make_cfg(32, 4, 4, 192, 64, families=["holdout"]),
                    make_cfg(8, 4, 4, 384, 128, mask="none", families=["holdout"]),
                    make_cfg(1, 32, 32, 384, 128, families=["holdout"]),
                    make_cfg(1, 64, 1, 384, 128, families=["holdout"]),
                    make_cfg(2, 4, 4, 192, 128, 64, families=["holdout"]),
                    make_cfg(1, 4, 4, 768, 64, families=["holdout"])]
    F["sentinel"] = [make_cfg(1, 1, 1, 256, 64, families=["sentinel"]),
                     make_cfg(1, 4, 4, 256, 128, families=["sentinel"]),
                     make_cfg(1, 64, 1, 256, 128, families=["sentinel"])]
    return F


def build_matrix(suites, expand_large=False):
    """Deduplicated union of the requested families, in first-seen order."""
    F = families()
    if expand_large:
        F["core"] += [make_cfg(4, H, H, 512, D, w, families=["core-large"])
                      for H in (1, 4) for D in (64, 128) for w in (512, 32)]
    out, seen = [], {}
    for s in suites:
        if s == "correctness":
            picks = F["core"] + F["shared-kv"]
        elif s == "all":
            picks = [c for k in ("core", "shared-kv", "bh", "batch", "sharing", "mask", "seq") for c in F[k]]
        else:
            picks = F[s]
        for c in picks:
            if c["name"] in seen:
                for f in c["families"]:
                    if f not in seen[c["name"]]["families"]:
                        seen[c["name"]]["families"].append(f)
            else:
                seen[c["name"]] = dict(c, families=list(c["families"]))
                out.append(seen[c["name"]])
    return out


def load_matrix_file(path):
    return [make_cfg(**{k: v for k, v in c.items() if k in ("B", "Hq", "Hkv", "N", "D", "w", "mask", "kv_repr", "families")})
            for c in json.load(open(path))]


# ---------------------------------------------------------------------------
# Inputs and the fp64 oracle
# ---------------------------------------------------------------------------

def window_mask(N, w, B, device):
    i = torch.arange(N, device=device)
    m = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - w)
    return m[None].expand(B, -1, -1).contiguous()


def make_inputs(cfg, seed, device="cuda", scale=1.0, dy_scale=1.0):
    """Canonical [B,Hq,N,D] bf16 draws; KV drawn once per KV head. The input
    seed is independent of the provider-order seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, Hq, Hkv, N, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["N"], cfg["D"]
    q = (torch.randn(B, Hq, N, D, generator=g) * scale).to(torch.bfloat16).to(device)
    kv = {n: (torch.randn(B, Hkv, N, D, generator=g) * scale).to(torch.bfloat16).to(device) for n in NAMES[1:]}
    dY = (torch.randn(B, Hq, N, D, generator=g) * dy_scale).to(torch.bfloat16).to(device)
    if cfg.get("kv_repr") == "repeated":
        kv = {n: expand_kv(t, Hq) for n, t in kv.items()}
        cfg = dict(cfg, Hkv=Hq)
    mask = None if cfg["mask"] == "none" else window_mask(N, cfg["w"], B, device)
    return {"Q": q, **kv, "dY": dY, "mask": mask, "cfg": cfg}


def input_hashes(inp):
    h = {}
    for n in NAMES + ["dY", "mask"]:
        t = inp[n]
        h[n] = None if t is None else hashlib.sha256(
            t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
    return h


def input_bytes(inp):
    return sum(inp[n].numel() * inp[n].element_size() for n in NAMES + ["dY", "mask"] if inp[n] is not None)


def expand_kv(t, Hq):
    """Grouped mapping: query head h reads KV head h // (Hq / Hkv)."""
    Hkv = t.size(1)
    if Hkv == Hq:
        return t
    return t.repeat_interleave(Hq // Hkv, dim=1).contiguous()


def reduce_kv_grad(g, Hkv, dim=1):
    """Sum a per-query-head KV gradient over each group back to Hkv heads."""
    Hq = g.size(dim)
    if Hq == Hkv:
        return g
    shape = list(g.shape)
    shape[dim:dim + 1] = [Hkv, Hq // Hkv]
    return g.float().view(*shape).sum(dim + 1).to(g.dtype)


def _forward_chunk(Q, R, S, Vr, Vs, mask, q0, q1):
    """fp64 Y and LSE for queries [q0, q1) of every (b, h) given; forms one
    [B,Hc,Nq,N,N] cube."""
    D = Q.shape[-1]
    x = torch.einsum("bhid,bhjd,bhkd->bhijk", Q[:, :, q0:q1], R, S) / math.sqrt(D)
    if mask is not None:
        vis = (mask[:, q0:q1, :, None] & mask[:, q0:q1, None, :])[:, None]
        x = x.masked_fill(~vis, float("-inf"))
    flat = x.flatten(3)
    lse = torch.logsumexp(flat, -1)
    p = torch.nan_to_num(torch.softmax(flat, -1), nan=0.0).reshape(x.shape)
    del x, flat
    Y = torch.einsum("bhijk,bhjd->bhikd", p, Vr)
    Y = torch.einsum("bhikd,bhkd->bhid", Y, Vs)
    return Y, lse


def oracle(inp, dense_budget=ORACLE_DENSE_BUDGET, chunk_budget=ORACLE_CHUNK_BUDGET):
    """Full-shape fp64 Y, LSE and the five gradients (KV gradients on the
    logical Hkv heads). Dense when B*Hq*N^3 fits the budget, otherwise
    streamed over head groups and query chunks with the gradients accumulated
    on the full-shape leaves; the two agree to fp64 rounding. Returns the
    tensors plus a record of how it was evaluated."""
    Q0 = inp["Q"]
    B, Hq, N, D = Q0.shape
    Hkv = inp["R"].size(1)
    leaves = {n: inp[n].double().requires_grad_(True) for n in NAMES}
    dY = inp["dY"].double()
    mask = inp["mask"]
    cube = B * Hq * N ** 3
    if cube <= dense_budget:
        Hc, Nq = Hq, N
        kind = "dense"
    else:
        Hc = max(1, min(Hq, chunk_budget // max(1, B * N ** 3)))
        Nq = N if Hc * B * N ** 3 <= chunk_budget else max(16, chunk_budget // max(1, B * N * N))
        Nq = min(N, Nq)
        kind = f"streamed(Hc={Hc},Nq={Nq})"
    Y = torch.zeros(B, Hq, N, D, dtype=torch.float64, device=Q0.device)
    lse = torch.zeros(B, Hq, N, dtype=torch.float64, device=Q0.device)
    chunks = 0
    for h0 in range(0, Hq, Hc):
        hs = slice(h0, min(Hq, h0 + Hc))
        kv_idx = torch.arange(hs.start, hs.stop, device=Q0.device) // (Hq // Hkv)
        for q0 in range(0, N, Nq):
            q1 = min(N, q0 + Nq)
            # Re-index per chunk: each backward frees the indexing graph.
            R, S, Vr, Vs = (leaves[n][:, kv_idx] for n in NAMES[1:])
            Yc, lc = _forward_chunk(leaves["Q"][:, hs], R, S, Vr, Vs, mask, q0, q1)
            Yc.backward(dY[:, hs, q0:q1])
            Y[:, hs, q0:q1] = Yc.detach()
            lse[:, hs, q0:q1] = lc.detach()
            del Yc, lc
            chunks += 1
    grads = {g: leaves[n].grad.detach() for g, n in zip(GRADS, NAMES)}
    return Y, lse, grads, {"kind": kind, "chunks": chunks, "full_shape": True}


def err_metrics(actual, ref):
    """Global relative L2 / normalized max error plus the worst (b, h) slice,
    so a broken head cannot hide inside a large aggregate."""
    a, r = actual.double(), ref.double()
    rel = ((a - r).norm() / max(r.norm().item(), 1e-12)).item()
    mx = ((a - r).abs().max() / max(1.0, r.abs().max().item())).item()
    out = {"rel_l2": rel, "max_norm": mx, "finite": bool(torch.isfinite(a).all())}
    if a.dim() >= 3:
        d = (a - r).flatten(2)
        rn = r.flatten(2).norm(dim=-1).clamp_min(1e-12)
        rel_bh = d.norm(dim=-1) / rn
        mx_bh = d.abs().amax(-1) / r.flatten(2).abs().amax(-1).clamp_min(1.0)
        i = int(rel_bh.argmax())
        out["bh_worst_rel_l2"] = rel_bh.flatten()[i].item()
        out["bh_worst_max_norm"] = mx_bh.flatten().max().item()
        out["bh_worst_index"] = [i // a.size(1), i % a.size(1)]
    return out


def within(name, m):
    """Global and worst-(b,h) metrics both inside the threshold."""
    rt, mt = TOL[name]
    return bool(m["finite"] and m["rel_l2"] <= rt and m["max_norm"] <= mt
                and m.get("bh_worst_rel_l2", 0.0) <= rt and m.get("bh_worst_max_norm", 0.0) <= mt)


def useful_flops(cfg):
    """4 D B Hq sum_i visible_j(i) visible_k(i): the useful GEMM work (score +
    value), excluding softmax and other scalar work."""
    N = cfg["N"]
    if cfg.get("mask", "window") == "none":
        pairs = N ** 3
    else:
        w = cfg["w"]
        pairs = sum(min(i + 1, w) ** 2 for i in range(N))
    return 4 * cfg["D"] * cfg["B"] * cfg["Hq"] * pairs


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
# Each provider converts the canonical inputs into its own layout (prepare),
# then exposes fwd / bwd on prepared state. Results come back canonical.
# caps: "fwd" (publishable forward), "bwd" (publishable backward), "fwd_state"
# (a forward whose Y/LSE are validated to gate the backward but not timed as a
# single-gather forward). lse_kind: "natural" (checked), "other" (reported
# only), None (no statistic).

class Provider:
    name = ""
    ops = ("fwd", "bwd", "fwd_bwd")
    caps = frozenset({"fwd", "bwd"})
    lse_kind = None
    layout = ""
    unavailable = None          # reason string when the provider cannot run
    config = {}

    def check(self, cfg):
        return None             # reason string when this config is unsupported

    def prepare(self, inp):
        raise NotImplementedError

    def fwd(self, st):
        raise NotImplementedError

    def bwd(self, st, out):
        raise NotImplementedError

    def bwd_reduce(self, st, grads):
        """Reduce replicated per-query-head KV gradients to the logical KV
        heads, in the provider's own layout (part of the full-operator backward)."""
        return grads

    def bwd_full(self, st, out):
        return self.bwd_reduce(st, self.bwd(st, out))

    def to_canonical_Y(self, st, out):
        return out[0]

    def lse(self, st, out):
        return None

    def to_canonical_grads(self, st, grads):
        return dict(zip(GRADS, grads))

    def saved_state_bytes(self, out):
        return sum(t.numel() * t.element_size() for t in out if isinstance(t, torch.Tensor))


def load_extension(path, tag):
    """Load a second build of the extension (e.g. the frozen cf1e258 .so) under
    its own module name so it can be timed next to the current one."""
    spec = importlib.util.spec_from_file_location(f"att3ntion_{tag}._cuda_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regime(cfg):
    return "none" if cfg["mask"] == "none" else ("full" if cfg["w"] >= cfg["N"] else "window")


class CudaSingleGather(Provider):
    """The single-gather CUDA kernels. `ext` is the extension module, `tiles`
    the sg_set_config() override applied before every call (0 = default), and
    `window` whether the causal/window metadata is passed so the kernels skip
    tiles outside it (V1+) or traverse every tile (V0). `tiles` may be one
    dict, or a dispatch table keyed "<D>" or "<D>:<regime>" (regime in full /
    window / none, from the mask metadata): the frozen dispatch rule a final
    candidate ships with."""
    layout = "[B,H,N,D] bf16 native; shared KV replicated to Hq heads"
    lse_kind = "natural"

    def __init__(self, name, ext, tiles=None, window=False, source="", ops=None):
        self.name, self.ext, self.tiles, self.window = name, ext, dict(tiles or {}), window
        self.config = {"extension": source, "tiles": self.tiles, "window_metadata": window}
        if ops:                      # tuning candidates time only the op their tiles touch
            self.ops = tuple(ops)
            self.caps = frozenset({"fwd"} if "bwd" not in ops else ({"fwd", "bwd"} if "fwd" in ops else {"fwd_state", "bwd"}))
        if ext is None:
            self.unavailable = "extension not importable"
        elif tiles and not hasattr(ext, "sg_set_config"):
            self.unavailable = "this build has no sg_set_config (frozen V0)"
        elif window and "window" not in getattr(getattr(ext, "single_gather_forward", None), "__doc__", "") :
            self.unavailable = "this build has no window metadata (frozen V0)"

    _applied = {}                # per extension: the tiles currently set, so back-to-back calls skip the host round trip

    def tiles_for(self, cfg):
        t = self.tiles
        if not any(isinstance(v, dict) for v in t.values()):
            return t
        D, r = str(cfg["D"]), regime(cfg)
        return t.get(f"{D}:{r}", t.get(D, t.get("default", {})))

    def _apply(self, tiles):
        if hasattr(self.ext, "sg_set_config") and CudaSingleGather._applied.get(id(self.ext)) != tiles:
            self.ext.sg_set_config({k: 0 for k in ("fwd_warps", "fwd_bk", "bwd_a_warps", "bwd_a_bk",
                                                   "bwd_r_warps", "bwd_r_bk", "bwd_dh")})
            if tiles:
                self.ext.sg_set_config(tiles)
            CudaSingleGather._applied[id(self.ext)] = dict(tiles)

    def prepare(self, inp):
        Hq = inp["Q"].size(1)
        st = {n: expand_kv(inp[n], Hq) for n in NAMES}
        st["dY"], st["mask"], st["Hkv"] = inp["dY"], inp["mask"], inp["R"].size(1)
        st["win"] = inp["cfg"]["w"] if (self.window and inp["mask"] is not None) else 0
        st["tiles"] = self.tiles_for(inp["cfg"])
        return st

    def fwd(self, st):
        self._apply(st["tiles"])
        if st["win"]:
            return self.ext.single_gather_forward(*[st[n] for n in NAMES], st["mask"], st["win"])
        return self.ext.single_gather_forward(*[st[n] for n in NAMES], st["mask"])

    def bwd(self, st, out):
        self._apply(st["tiles"])
        if st["win"]:
            return self.ext.single_gather_backward(st["dY"], *[st[n] for n in NAMES], *out, st["mask"], st["win"])
        return self.ext.single_gather_backward(st["dY"], *[st[n] for n in NAMES], *out, st["mask"])

    def bwd_reduce(self, st, grads):
        g = list(grads)
        return [g[0]] + [reduce_kv_grad(t, st["Hkv"]) for t in g[1:]]

    def lse(self, st, out):
        m, l = out[1], out[2]
        return torch.where(l > 0, m.double() + torch.log(l.double().clamp_min(1e-300)),
                           torch.full_like(m.double(), float("-inf")))

    def dispatch(self):
        return self.ext.sg_dispatch() if hasattr(self.ext, "sg_dispatch") else None


class CudaShared(Provider):
    """Experimental shared-KV prototype: the four KV tensors stay [B,1,N,128] and are
    read in place (no `expand_kv`), the backward returns KV gradients already reduced
    over heads, so `bwd_reduce` is the identity. fwd_group / rs_group pick the native
    (1) or grouped (2, 4) kernels for the forward and the R/S-owned passes."""
    layout = "Q [B,Hq,N,D] bf16, KV [B,1,N,D] bf16 read in place (no replication)"
    lse_kind = "natural"

    def __init__(self, name, ext, fwd_group=1, rs_group=1, source=""):
        self.name, self.ext, self.fwd_group, self.rs_group = name, ext, fwd_group, rs_group
        self.config = {"extension": source, "fwd_group": fwd_group, "rs_group": rs_group, "kv": "native Hkv=1"}
        if ext is None:
            self.unavailable = "extension not importable"
        elif not hasattr(ext, "single_gather_shared_forward"):
            self.unavailable = "this build has no single_gather_shared_* entry points"

    def check(self, cfg):
        if cfg["Hkv"] != 1 or cfg.get("kv_repr") == "repeated":
            return "shared prototype needs Hkv = 1"
        if cfg["D"] != 128:
            return "shared prototype is D = 128 only"
        if cfg["mask"] != "window" or cfg["w"] != 32:
            return "shared prototype is window = 32 only"
        if cfg["Hq"] % max(self.fwd_group, self.rs_group):
            return f"Hq must be divisible by the head group {max(self.fwd_group, self.rs_group)}"
        return None

    def prepare(self, inp):
        st = {n: inp[n] for n in NAMES}          # KV stays [B,1,N,D]
        st["dY"], st["mask"], st["Hkv"] = inp["dY"], inp["mask"], inp["R"].size(1)
        return st

    def fwd(self, st):
        return self.ext.single_gather_shared_forward(*[st[n] for n in NAMES], st["mask"], 32, self.fwd_group)

    def bwd(self, st, out):
        return self.ext.single_gather_shared_backward(st["dY"], *[st[n] for n in NAMES], *out, st["mask"], 32, self.rs_group)

    def lse(self, st, out):
        m, l = out[1], out[2]
        return torch.where(l > 0, m.double() + torch.log(l.double().clamp_min(1e-300)),
                           torch.full_like(m.double(), float("-inf")))

    def dispatch(self):
        return self.ext.sg_dispatch()


class CudaLegacyUnspecialized(Provider):
    """Ablation: the legacy three-gather backward (Bwd_gather_tc<BWD_ALL> x3)
    fed zero cotangents for the two unused gathers. Its forward is the legacy
    Q-only forward, which also fabricates the absent gathers' stats and zero
    outputs; that state is prepared outside the backward timing and its cost
    is reported as `legacy_state_fwd`, not as a single-gather forward. Its Y
    and LSE are validated to gate the backward."""
    name = "cuda_unspecialized_legacy"
    ops = ("bwd", "legacy_state_fwd")
    caps = frozenset({"fwd_state", "bwd"})
    lse_kind = "natural"
    layout = "[B,H,N,D] bf16; 9-input legacy API with zero Vq/V2"

    def __init__(self, ext):
        self.ext = ext
        if ext is None:
            self.unavailable = "extension not importable"

    def prepare(self, inp):
        Hq = inp["Q"].size(1)
        st = {n: expand_kv(inp[n], Hq) for n in NAMES}
        st["zero"] = torch.zeros_like(st["Q"])
        st["dY"], st["mask"], st["Hkv"] = inp["dY"], inp["mask"], inp["R"].size(1)
        st["N"] = st["Q"].size(2)
        return st

    def fwd(self, st):
        z, N = st["zero"], st["N"]
        return self.ext.forward(st["Q"], st["R"], st["S"], z, z, st["Vr"], z, st["Vs"], z,
                                0.0, N, N, N, st["mask"], 1)

    def bwd(self, st, out):
        z = st["zero"]
        g = self.ext.backward(st["dY"], z, z, z, z, z, st["Q"], st["R"], st["S"], z, z,
                              st["Vr"], z, st["Vs"], z, *out[6:12], 0.0, st["mask"],
                              out[0], out[1], out[2])
        return g[0], g[1], g[2], g[5], g[7]

    def bwd_reduce(self, st, grads):
        g = list(grads)
        return [g[0]] + [reduce_kv_grad(t, st["Hkv"]) for t in g[1:]]

    def lse(self, st, out):
        m, l = out[6], out[7]
        return torch.where(l > 0, m.double() + torch.log(l.double().clamp_min(1e-300)),
                           torch.full_like(m.double(), float("-inf")))


PAPER_DEFAULT = dict(BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64, num_warps=4, num_stages=1)


class PaperTriton(Provider):
    """Appendix listings via benchmarks/paper_simplicial.py. Layout [b,s,k,h].
    fwd_cfg / bwd_cfg are the Listing 1 / Listing 2 launch constants (the
    paper's autotune candidates, passed explicitly)."""
    layout = "[b,s,k,h] bf16 (permute of canonical); shared KV replicated to Hq heads"
    lse_kind = "natural"
    bwd_kind = "general"

    MANY_CTAS = 64          # B*Hq at or above this uses the "many" constants (a disclosed selection rule)

    def __init__(self, name="paper_triton", fwd_cfg=None, bwd_cfg=None, ops=None):
        self.name = name
        self.fwd_cfg = self._table(fwd_cfg)
        self.bwd_cfg = self._table(bwd_cfg)
        self.config = {"forward": self.fwd_cfg, "backward": self.bwd_cfg, "backward_kind": self.bwd_kind,
                       "selection_rule": f"'many' when B*Hq >= {self.MANY_CTAS}, else 'few'"}
        if ops:                      # tuning candidates time only the op their constants touch
            self.ops = tuple(ops)
            self.caps = frozenset({"fwd"} if "bwd" not in ops else ({"fwd", "bwd"} if "fwd" in ops else {"fwd_state", "bwd"}))
        try:
            import paper_simplicial
            self.ps = paper_simplicial
        except Exception as e:      # noqa: BLE001
            self.unavailable = f"paper_simplicial import failed: {e!r}"

    @staticmethod
    def _table(cfg):
        """One launch configuration, or {"few": {...}, "many": {...}} selected per call by B*Hq."""
        if cfg and ("few" in cfg or "many" in cfg):
            return {k: dict(PAPER_DEFAULT, **v) for k, v in cfg.items()}
        return dict(PAPER_DEFAULT, **(cfg or {}))

    def _pick(self, table, cfg):
        if "few" not in table and "many" not in table:
            return table
        key = "many" if cfg["B"] * cfg["Hq"] >= self.MANY_CTAS else "few"
        return table.get(key, table.get("few", table.get("many")))

    def check(self, cfg):
        if cfg["mask"] == "none":
            return "paper kernels are causal (w = N); unmasked needs a validated source adaptation"
        return None

    def prepare(self, inp):
        Hq = inp["Q"].size(1)
        to = lambda t: t.permute(0, 2, 1, 3).contiguous()
        st = {n: to(expand_kv(inp[n], Hq)) for n in NAMES}
        st["dY"] = to(inp["dY"])
        st["w"], st["Hkv"] = inp["cfg"]["w"], inp["R"].size(1)
        st["fwd_cfg"], st["bwd_cfg"] = self._pick(self.fwd_cfg, inp["cfg"]), self._pick(self.bwd_cfg, inp["cfg"])
        return st

    def fwd(self, st):
        return self.ps.paper_fwd(*[st[n] for n in NAMES], st["w"], st["w"], **st["fwd_cfg"])

    def bwd(self, st, out):
        fn = self.ps.paper_bwd_general if self.bwd_kind == "general" else self.ps.paper_bwd_small_w2
        return fn(*[st[n] for n in NAMES], out[0], out[1], st["dY"], st["w"], st["w"], **st["bwd_cfg"])

    def bwd_reduce(self, st, grads):
        g = list(grads)
        return [g[0]] + [reduce_kv_grad(t, st["Hkv"], dim=2) for t in g[1:]]

    def to_canonical_Y(self, st, out):
        return out[0].permute(0, 2, 1, 3).contiguous()

    def lse(self, st, out):
        return out[1].double()

    def to_canonical_grads(self, st, grads):
        return {g: t.permute(0, 2, 1, 3).contiguous() for g, t in zip(GRADS, grads)}


class PaperTritonSmallW2(PaperTriton):
    """Listing 3 (no atomics) for dQ/dK2/dV2 + Listing 2 for dK1/dV1; w must be
    32 (BLOCK_SIZE_KV2 = BLOCK_SIZE_Q + w2). Its forward is the ordinary Listing 1
    (a real, validated single-gather forward, so the composite's complete-operation
    samples are eligible); it is not timed separately here because `paper_triton`
    already publishes that number."""
    ops = ("bwd", "fwd_bwd")
    caps = frozenset({"fwd", "bwd"})
    bwd_kind = "small_w2"

    def __init__(self, name="paper_triton_listing3_w32", fwd_cfg=None, bwd_cfg=None):
        super().__init__(name, fwd_cfg, bwd_cfg)

    def check(self, cfg):
        if cfg["mask"] == "none":
            return "paper kernels are causal (w = N); unmasked needs a validated source adaptation"
        if cfg["w"] != 32:
            return "Listing 3 needs w2 = BLOCK_SIZE_KV2 - BLOCK_SIZE_Q = 32"
        return None


class FbgemmTritonFwd(Provider):
    """FBGEMM simplicial.ops.triton.fwd GQA-packed forward (kv heads = 1, all
    query heads form the M tile). Launched directly so no bias path is touched:
    the upstream `triton_fwd` wrapper's `if not k2_bias` would re-enable
    bias = 1/head_dim for a zero argument. This provider is forward only:
    its stock saved statistic is not a natural-log LSE. FbgemmTraining can
    compose backward with an explicitly identified natural-LSE source copy."""
    name = "fbgemm_triton_fwd"
    ops = ("fwd",)
    caps = frozenset({"fwd"})
    lse_kind = "other"
    layout = "q [b,s,k,h] bf16, kv [b,s,1,h] bf16 (no replication)"

    def __init__(self, fbgemm_path):
        if fbgemm_path and fbgemm_path not in sys.path:
            sys.path.insert(0, fbgemm_path)
        try:
            from simplicial.ops.triton import fwd as fb
            self.fb = fb
        except Exception as e:      # noqa: BLE001
            self.unavailable = f"fbgemm simplicial import failed: {e!r}"

    def check(self, cfg):
        if cfg["mask"] == "none":
            return "FBGEMM forward is causal (w = N); unmasked needs a validated source adaptation"
        if cfg["Hkv"] != 1 or cfg.get("kv_repr") == "repeated":
            return "GQA-packed kernel assumes one shared KV head"
        if cfg["Hq"] < 16:
            return "BLOCK_SIZE_Q = Hq must be >= 16 for tl.dot"
        return None

    def prepare(self, inp):
        to = lambda t: t.permute(0, 2, 1, 3).contiguous()
        st = {n: to(inp[n]) for n in NAMES}
        st["w"] = inp["cfg"]["w"]
        return st

    def fwd(self, st):
        q, k1, k2, v1, v2 = (st[n] for n in NAMES)
        bs, seq_len, num_heads, head_dim = q.shape
        w = st["w"]
        output = torch.empty_like(q)
        m = torch.empty((bs, num_heads, seq_len), dtype=torch.float32, device=q.device)
        sm_scale = 1.44269504 * head_dim ** -0.5
        s = lambda t: (t.stride(0), t.stride(1), t.stride(2), t.stride(3))
        self.fb._gqa_pack_fwd_kernel[(seq_len, bs)](
            q, k1, k2, v1, v2, output, m, bs, seq_len, num_heads, w, w,
            *s(q), *s(k1), *s(k2), *s(v1), *s(v2), *s(output),
            m.stride(0), m.stride(1), m.stride(2),
            HEAD_DIM=head_dim, INPUT_PRECISION="tf32", SM_SCALE=sm_scale,
            K2_BIAS=0.0, V2_BIAS=0.0, BLOCK_SIZE_Q=num_heads)
        return output, m

    def to_canonical_Y(self, st, out):
        return out[0].permute(0, 2, 1, 3).contiguous()

    def lse(self, st, out):
        return out[1].double()      # reported, not trusted: log2-domain max + ln(l)


class TlxFwd(FbgemmTritonFwd):
    """Released FBGEMM TLX forward; no competitor code is used by our CUDA kernels.

    The stock saved statistic mixes log bases, so it is reported but is not
    claimed to be a natural LSE. A training experiment must explicitly load
    and identify a corrected external source copy before changing that claim.
    """
    name = "tlx_fwd_ws"
    ops = ("fwd",)
    caps = frozenset({"fwd"})

    def __init__(self, fbgemm_path, variant="ws"):
        if variant not in ("ws", "ws_pipelined", "ws_pingpong"):
            raise ValueError(f"Unknown TLX variant: {variant}")
        self.name = f"tlx_fwd_{variant}"
        self.config = {"variant": variant, "source": fbgemm_path, "saved_lse": "upstream mixed log bases"}
        if fbgemm_path and fbgemm_path not in sys.path:
            sys.path.insert(0, fbgemm_path)
        try:
            import triton.language.extra.tlx  # noqa: F401
            from importlib import import_module
            self.fb = import_module(f"simplicial.ops.tlx.fwd_{variant}")
            self.forward = getattr(self.fb, f"tlx_fwd_{variant}")
        except Exception as e:      # noqa: BLE001
            self.unavailable = f"TLX not importable ({e!r}); needs the TLX Triton fork"

    def check(self, cfg):
        reason = super().check(cfg)
        if reason:
            return reason
        if cfg["Hq"] not in (64, 128):
            return "released TLX consumer dispatch supports Hq = 64 or 128"
        if self.config["variant"] == "ws_pingpong" and cfg["Hq"] != 128:
            return "TLX ping-pong requires two consumer groups (Hq=128); Hq=64 triggers an illegal barrier arrive"
        if cfg["D"] != 128 or cfg["w"] not in (16, 32, 64, 128):
            return "TLX adapter validation scope: D = 128, w = 16/32/64/128"
        return None

    def fwd(self, st):
        return self.forward(*[st[n] for n in NAMES], st["w"], st["w"])


class FbgemmTraining(Provider):
    """Natural-LSE FBGEMM forward + released Triton backward, including KV sum.

    The backward uses Q's batch/head offsets for every input. It therefore
    requires physically replicated KV tensors with Q's layout. Preparation
    holds those copies; API timing also includes making them. Backward timing
    always includes reducing the four KV gradients to the logical KV head.
    """
    lse_kind = "natural"
    layout = "forward shared KV; backward replicated [B,N,H,D] KV with head reduction"

    def __init__(self, forward_provider):
        if forward_provider.lse_kind != "natural":
            raise ValueError("Training requires an explicitly identified natural-LSE forward source")
        self.forward_provider = forward_provider
        self.name = forward_provider.name + "_triton_bwd"
        self.config = dict(forward_provider.config, backward="released FBGEMM Triton; w2=32")
        self.unavailable = forward_provider.unavailable
        if not self.unavailable:
            from simplicial.ops.triton.bwd import triton_bwd
            self.backward = triton_bwd

    def check(self, cfg):
        reason = self.forward_provider.check(cfg)
        return reason or ("released FBGEMM backward requires w2 = 32" if cfg["w"] != 32 else None)

    def prepare(self, inp):
        st = self.forward_provider.prepare(inp)
        H = st["Q"].shape[2]
        st["bwd_kv"] = [st[n].expand(-1, -1, H, -1).contiguous() for n in NAMES[1:]]
        st["dY"] = inp["dY"].permute(0, 2, 1, 3).contiguous()
        st["Hkv"] = inp["R"].shape[1]
        return st

    def fwd(self, st):
        return self.forward_provider.fwd(st)

    def bwd(self, st, out):
        return self.backward(st["Q"], *st["bwd_kv"], out[0], st["dY"], out[1], st["w"], st["w"])

    def bwd_reduce(self, st, grads):
        return [grads[0]] + [reduce_kv_grad(g, st["Hkv"], dim=2) for g in grads[1:]]

    def to_canonical_Y(self, st, out):
        return self.forward_provider.to_canonical_Y(st, out)

    def lse(self, st, out):
        return self.forward_provider.lse(st, out)

    def to_canonical_grads(self, st, grads):
        return {n: g.permute(0, 2, 1, 3).contiguous() for n, g in zip(GRADS, grads)}


def cuda_candidates(spec_path, v0_path):
    """CUDA candidates from a JSON list of {name, ext: current|v0|<path>, tiles,
    window}; the default is the frozen V0 (dense, default tiles), the current
    build without window metadata (build-drift check) and the current build
    with it."""
    if spec_path:
        specs = json.load(open(spec_path))
    else:
        specs = [{"name": "cuda_v0", "ext": "v0", "tiles": {}, "window": False},
                 {"name": "cuda_v0b_dense", "ext": "current", "tiles": {}, "window": False},
                 {"name": "cuda_v1_window", "ext": "current", "tiles": {}, "window": True}]
    exts = {"current": (ck, getattr(ck, "__file__", None))}
    out = []
    for s in specs:
        key = s.get("ext", "current")
        if key not in exts:
            path = v0_path if key == "v0" else key
            try:
                exts[key] = (load_extension(path, key.replace("/", "_").replace(".", "_")), path)
            except Exception as e:      # noqa: BLE001
                exts[key] = (None, f"{path}: {e!r}")
        ext, src = exts[key]
        if isinstance(src, str) and os.path.exists(src):
            s = dict(s, extension_sha256=sha256_file(src))     # binary identity in the manifest
        ext, src = exts[key]
        if s.get("kind") == "shared":
            p = CudaShared(s["name"], ext, s.get("fwd_group", 1), s.get("rs_group", 1), source=str(src))
        else:
            p = CudaSingleGather(s["name"], ext, s.get("tiles"), s.get("window", False), source=str(src), ops=s.get("ops"))
        p.config = dict(p.config, extension_sha256=s.get("extension_sha256"), note=s.get("note"))
        if ext is None:
            p.unavailable = f"extension {key} not loadable ({src})"
        out.append(p)
    return out


def paper_candidates(spec_path):
    """Paper providers: the historical fixed configuration, the printed
    Listing 1 autotune pick (Q64/KV32), the Listing 3 path, and any tuned
    entries from a JSON list of {name, fwd, bwd}."""
    out = [PaperTriton("paper_triton"),
           PaperTriton("paper_triton_l1print", fwd_cfg=dict(BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=32)),
           PaperTritonSmallW2()]
    if spec_path:
        for s in json.load(open(spec_path)):
            out.append(PaperTriton(s["name"], s.get("fwd"), s.get("bwd"), ops=s.get("ops")))
    return out


def all_providers(args):
    return (cuda_candidates(args.candidates, args.v0_ext)
            + [CudaLegacyUnspecialized(ck)]
            + paper_candidates(args.paper_candidates)
            + [FbgemmTritonFwd(args.fbgemm_path)]
            + [TlxFwd(args.fbgemm_path, v) for v in ("ws", "ws_pipelined", "ws_pingpong")])


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------

def _oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _status_of(e):
    if _oom(e):
        return "oom"
    name = type(e).__name__
    if "Compilation" in name or "compile" in str(e).lower():
        return "compile_error"
    if "invalid configuration" in str(e).lower() or "invalid argument" in str(e).lower():
        return "invalid_launch"
    return "error"


def gate(rec, op):
    """Whether `op` of a provider is publishable from its correctness record.
    fwd: Y and (if claimed natural) LSE. bwd: its own forward state validated
    plus every gradient and the repeat check. Combined/API ops need both."""
    if rec.get("status") != "pass" and rec.get("status") != "fail":
        return False
    fwd_ok = bool(rec.get("fwd_ok")) and (rec.get("lse_ok") is not False) and rec.get("full_shape_ok", True)
    bwd_ok = fwd_ok and bool(rec.get("bwd_ok")) and rec.get("repeat_ok", True) is not False
    if op in ("fwd", "api_fwd"):
        return fwd_ok and "fwd" in rec.get("caps", ())
    if op in ("bwd", "bwd_prepared"):
        return bwd_ok and "bwd" in rec.get("caps", ())
    if op in ("fwd_bwd", "api_fwd_bwd"):
        return bwd_ok and "fwd" in rec.get("caps", ()) and "bwd" in rec.get("caps", ())
    return False       # legacy_state_fwd and other component metrics carry no ratio


def check_provider(p, inp, ref):
    """Run p on the inputs, compare every output/gradient against the fp64
    oracle at the full shape, check the repeat-call consistency of the
    backward, and account memory. Returns (record, state, forward_out)."""
    Y_ref, lse_ref, g_ref, _ = ref
    rec = {"provider": p.name, "caps": sorted(p.caps), "lse_kind": p.lse_kind, "status": "pass"}
    torch.cuda.synchronize()
    a0 = torch.cuda.memory_allocated()
    st = p.prepare(inp)
    torch.cuda.synchronize()
    rec["prepared_bytes"] = torch.cuda.memory_allocated() - a0
    # Prepared KV storage actually held (catches expand(...).contiguous() copies): bytes of the
    # KV tensors the provider will read, whatever their displayed shape.
    rec["prepared_kv_bytes"] = sum(st[n].untyped_storage().nbytes() for n in NAMES[1:] if isinstance(st.get(n), torch.Tensor))
    out = p.fwd(st)
    torch.cuda.synchronize()
    rec["saved_state_bytes"] = p.saved_state_bytes(out)
    ok = True
    if "fwd" in p.caps or "fwd_state" in p.caps:
        Y = p.to_canonical_Y(st, out)
        rec["Y"] = err_metrics(Y, Y_ref)
        rec["fwd_ok"] = within("Y", rec["Y"])
        dead = lse_ref == float("-inf")
        if dead.any():
            rec["dead_rows_zero"] = bool(Y[dead].abs().max().item() == 0.0)
            rec["fwd_ok"] = rec["fwd_ok"] and rec["dead_rows_zero"]
        lse = p.lse(st, out)
        if lse is not None:
            live = ~dead
            rec["lse_dead_rows_match"] = bool(torch.equal(dead, lse == float("-inf")))
            rec["lse_max_err"] = (lse[live] - lse_ref[live]).abs().max().item() if live.any() else 0.0
            if p.lse_kind == "natural":
                rec["lse_ok"] = rec["lse_dead_rows_match"] and rec["lse_max_err"] <= LSE_MAX
            else:
                rec["lse_ok"] = None            # reported, not a natural-log LSE claim
        ok &= rec["fwd_ok"] and rec.get("lse_ok") is not False
    if "bwd" in p.caps:
        grads = p.to_canonical_grads(st, p.bwd_full(st, out))
        torch.cuda.synchronize()
        gok = True
        for g in GRADS:
            rec[g] = err_metrics(grads[g], g_ref[g])
            gok &= within(g, rec[g])
        rec["bwd_ok"] = bool(gok)
        # Repeated call on the same state: stale accumulation shows up as a
        # doubled gradient (rel L2 ~ 1); atomics may differ in the last bits.
        grads2 = p.to_canonical_grads(st, p.bwd_full(st, out))
        torch.cuda.synchronize()
        rec["repeat_max_rel_l2"] = max(err_metrics(grads2[g], grads[g])["rel_l2"] for g in GRADS)
        rec["repeat_ok"] = rec["repeat_max_rel_l2"] <= REPEAT_MAX_REL_L2
        ok &= rec["bwd_ok"] and rec["repeat_ok"]
        del grads, grads2
    rec["full_shape_ok"] = True      # the oracle ran at the full shape
    rec["status"] = "pass" if ok else "fail"
    return rec, st, out


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_op(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ev, wall = [], []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1e3)
        ev.append(s.elapsed_time(e))
    return ev, wall, {"peak_incremental_bytes": torch.cuda.max_memory_allocated() - base,
                      "alloc_before_bytes": base, "reserved_bytes": torch.cuda.memory_reserved()}


def op_callable(p, op, inp, st, out):
    if op == "fwd":
        return lambda: p.fwd(st)
    if op == "bwd":
        return lambda: p.bwd_full(st, out)
    if op == "bwd_prepared":
        return lambda: p.bwd(st, out)
    if op == "fwd_bwd":
        return lambda: p.bwd_full(st, p.fwd(st))
    if op == "legacy_state_fwd":
        return lambda: p.fwd(st)
    if op == "api_fwd":
        return lambda: p.to_canonical_Y(*(lambda s: (s, p.fwd(s)))(p.prepare(inp)))
    if op == "api_fwd_bwd":
        def run():
            s = p.prepare(inp)
            o = p.fwd(s)
            return p.to_canonical_grads(s, p.bwd_full(s, o))
        return run
    raise ValueError(op)


def provider_ops(p, cfg):
    ops = list(p.ops)
    if "bwd" in ops and cfg["Hkv"] < cfg["Hq"] and cfg.get("kv_repr") != "repeated":
        ops.insert(ops.index("bwd") + 1, "bwd_prepared")
    if "fwd" in p.ops:
        ops.append("api_fwd")
    if "fwd_bwd" in p.ops:
        ops.append("api_fwd_bwd")
    return ops


def summarize(xs):
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"median": statistics.median(xs), "p25": q(0.25), "p75": q(0.75), "min": xs[0], "n": len(xs)}


def calibrate_repeats(estimates_ms, target_ms, lo=5, hi=200):
    """Repeat count for one comparison from the providers' warm estimates:
    about target_ms per round at the providers' geometric-mean latency, same
    count for every provider in the comparison."""
    est = [e for e in estimates_ms if e and e > 0]
    if not est:
        return lo
    gm = math.exp(sum(math.log(e) for e in est) / len(est))
    return int(max(lo, min(hi, round(target_ms / gm))))


def sentinel_ms(iters=20):
    """A fixed matmul timed the same way, run before/after every configuration:
    drift beyond 5% between readings flags a contaminated session."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    fn = lambda: a @ a
    ev, _, _ = time_op(fn, 20, iters)
    del a
    return statistics.median(ev)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def smi(fields):
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:      # noqa: BLE001
        return f"unavailable: {e!r}"


def gpu_state():
    return {"time": datetime.now().isoformat(),
            "smi": smi("utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw"),
            "competing_processes": subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip() or "none"}


def git_info():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        diff = subprocess.run(["git", "diff", "HEAD", "--stat"], cwd=ROOT, capture_output=True, text=True).stdout
        return {"rev": rev, "dirty": [l for l in dirty.splitlines() if l.strip()], "diff_stat": diff.strip()}
    except Exception as e:      # noqa: BLE001
        return {"error": repr(e)}


def environment(args):
    props = torch.cuda.get_device_properties(0)
    files = ["cuda/forward.cu", "cuda/backward.cu", "cuda/common.cuh", "cpp/cuda_bindings.cpp",
             "cpp/cuda_bindings.h", "att3ntion/_single_gather.py", "benchmarks/bench_single_gather.py",
             "benchmarks/paper_simplicial.py", "tests/test_single_gather.py"]
    hashes = {f: sha256_file(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))}
    fb = os.path.join(args.fbgemm_path or "", "simplicial/ops/triton/fwd.py")
    if os.path.exists(fb):
        hashes["fbgemm/simplicial/ops/triton/fwd.py"] = sha256_file(fb)
    commit = os.path.join(args.fbgemm_path or "", "SOURCE_COMMIT.txt")
    try:
        import triton
        triton_v = triton.__version__
    except Exception:       # noqa: BLE001
        triton_v = None
    ext = getattr(ck, "__file__", None)
    uuid = smi("gpu_uuid,name,driver_version")
    return {
        "gpu": props.name, "gpu_uuid_name_driver": uuid, "sm_count": props.multi_processor_count,
        "smem_optin": props.shared_memory_per_block_optin, "torch": torch.__version__,
        "cuda": torch.version.cuda, "triton": triton_v, "python": platform.python_version(),
        "extension": ext, "extension_sha256": sha256_file(ext) if ext and os.path.exists(ext) else None,
        "v0_extension": args.v0_ext,
        "v0_extension_sha256": sha256_file(args.v0_ext) if os.path.exists(args.v0_ext) else None,
        "source_sha256": hashes, "git": git_info(),
        "fbgemm_commit": open(commit).read().strip() if os.path.exists(commit) else None,
        "hostname": platform.node(), "gpu_state_start": gpu_state(),
    }


def dispatch_trace(p, inp):
    from torch.profiler import ProfilerActivity, profile
    st = p.prepare(inp)
    out = p.fwd(st)
    if "bwd" in p.caps:
        p.bwd_full(st, out)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = p.fwd(st)
        if "bwd" in p.caps:
            p.bwd_full(st, out)
        torch.cuda.synchronize()
    kernels = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0) > 0:
            kernels[e.key] = {"count": e.count, "device_us": round(getattr(e, "self_device_time_total", 0.0), 1)}
    return {k: v for k, v in kernels.items() if v["count"] > 0}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def run_session(args, out_dir, matrix, providers, session):
    order_rng = random.Random(args.order_seed + 1000 * session)
    manifest = {
        "run_id": os.path.basename(out_dir), "suite": args.suite, "session": session,
        "command": " ".join(sys.argv), "args": vars(args), "environment": environment(args),
        "started": datetime.now().isoformat(),
        "providers": {p.name: {"ops": list(p.ops), "caps": sorted(p.caps), "lse_kind": p.lse_kind,
                               "layout": p.layout, "unavailable": p.unavailable, "config": p.config}
                      for p in providers},
        "matrix": matrix, "rounds": [], "repeats": {}, "sentinel_ms": [], "input_hashes": {}, "notes": [
            "windows: a query i sees keys j with i-w < j <= i on both axes; w=N is full causal; mask=none is unmasked",
            "device latency: CUDA events around the warmed call, output allocation inside the call",
            "bwd: full-operator backward incl. delta reduction, bf16 conversion and KV-gradient reduction to Hkv; "
            "bwd_prepared: the reduction-free component (shared KV only)",
            "api latency: synchronized wall clock incl. layout conversion / KV replication / grad reduction",
            "cuda_unspecialized_legacy: backward only; legacy_state_fwd is its Q-only three-gather forward, not a single-gather forward",
            "fbgemm_triton_fwd: forward only; saved statistic is log2-domain max + ln(l), not natural-log LSE (reported, not gated)",
            "cuda providers apply sg_set_config() when their tiles differ from the last applied (host-side, microseconds; inside wall time)",
            "sentinel_ms: 2048^2 bf16 matmul median before/after each configuration; >5% drift flags contamination",
        ],
    }
    print(f"[{manifest['run_id']}] session {session} gpu={manifest['environment']['gpu']} ext={getattr(ck, '__file__', None)}")
    for p in providers:
        if p.unavailable:
            print(f"  provider {p.name}: UNAVAILABLE ({p.unavailable})")

    correctness, traces, summary = {}, {}, []
    samples_f = open(os.path.join(out_dir, "samples.jsonl"), "w")
    manifest["sentinel_ms"].append({"at": "session_start", "ms": sentinel_ms()})
    t_session = time.time()

    for cfg in matrix:
        t_cfg = time.time()
        inp = make_inputs(cfg, args.seed)
        manifest["input_hashes"][cfg["name"]] = input_hashes(inp)
        try:
            ref = oracle(inp)
        except Exception as e:      # noqa: BLE001
            correctness[cfg["name"]] = {"oracle": {"error": repr(e), "status": _status_of(e)}, "providers": {}}
            print(f"  {cfg['name']:36s} ORACLE {_status_of(e)}: {e!r}")
            torch.cuda.empty_cache()
            continue
        cres = {"oracle": ref[3], "input_bytes": input_bytes(inp), "providers": {}}
        active = []
        for p in providers:
            reason = p.unavailable or p.check(cfg)
            if reason:
                cres["providers"][p.name] = {"provider": p.name, "status": "unavailable" if p.unavailable else "unsupported",
                                             "reason": reason, "caps": sorted(p.caps)}
                continue
            try:
                r, st, out = check_provider(p, inp, ref)
                if isinstance(p, (CudaSingleGather, CudaShared)):
                    r["dispatch"] = p.dispatch()
                traces.setdefault(cfg["name"], {})[p.name] = dispatch_trace(p, inp)
                cres["providers"][p.name] = r
                active.append((p, st, out))
                flag = r["status"].upper()
                print(f"  {cfg['name']:36s} {p.name:26s} {flag}  " + " ".join(
                    f"{k}={r[k]['rel_l2']:.4f}/{r[k]['max_norm']:.4f}" for k in ["Y"] + GRADS if k in r)
                    + (f" lse={r['lse_max_err']:.4f}" if "lse_max_err" in r else ""))
            except Exception as e:      # noqa: BLE001
                cres["providers"][p.name] = {"provider": p.name, "status": _status_of(e), "error": repr(e),
                                             "caps": sorted(p.caps)}
                print(f"  {cfg['name']:36s} {p.name:26s} {_status_of(e).upper()} {e!r}")
                torch.cuda.empty_cache()
        cres["oracle_seconds"] = round(time.time() - t_cfg, 2)
        correctness[cfg["name"]] = cres
        if args.suite == "correctness" or args.rounds == 0:
            del ref
            torch.cuda.empty_cache()
            continue

        # Calibrate one repeat count per comparison (config, op) from warm estimates.
        est = {}
        for p, st, out in active:
            for op in provider_ops(p, cfg):
                fn = op_callable(p, op, inp, st, out)
                try:
                    ev, _, _ = time_op(fn, 3, 3)
                    est.setdefault(op, []).append(statistics.median(ev))
                except Exception as e:      # noqa: BLE001
                    cres["providers"][p.name].setdefault("timing_errors", []).append({"op": op, "error": repr(e)})
        repeats = {op: (args.iters if args.iters else calibrate_repeats(v, args.target_round_ms)) for op, v in est.items()}
        manifest["repeats"][cfg["name"]] = repeats
        manifest["sentinel_ms"].append({"at": f"before:{cfg['name']}", "ms": sentinel_ms()})

        order = [p.name for p, _, _ in active]
        for rnd in range(args.rounds):
            order_rng.shuffle(order)
            manifest["rounds"].append({"config": cfg["name"], "round": rnd, "order": list(order)})
            for pname in order:
                p, st, out = next(t for t in active if t[0].name == pname)
                for op in provider_ops(p, cfg):
                    if op not in repeats:
                        continue
                    fn = op_callable(p, op, inp, st, out)
                    try:
                        ev, wall, mem = time_op(fn, args.warmup if rnd == 0 else 2, repeats[op])
                    except Exception as e:      # noqa: BLE001
                        cres["providers"][p.name].setdefault("timing_errors", []).append(
                            {"op": op, "round": rnd, "error": repr(e)})
                        torch.cuda.empty_cache()
                        continue
                    for i, (a, b) in enumerate(zip(ev, wall)):
                        samples_f.write(json.dumps({"config": cfg["name"], "provider": pname, "op": op,
                                                    "session": session, "round": rnd, "iter": i,
                                                    "event_ms": a, "wall_ms": b}) + "\n")
                    summary.append({"config": cfg["name"], "provider": pname, "op": op, "round": rnd, **mem})
        manifest["sentinel_ms"].append({"at": f"after:{cfg['name']}", "ms": sentinel_ms()})
        samples_f.flush()
        del ref, active
        torch.cuda.empty_cache()

    manifest["sentinel_ms"].append({"at": "session_end", "ms": sentinel_ms()})
    s_ms = [x["ms"] for x in manifest["sentinel_ms"]]
    manifest["sentinel_drift"] = (max(s_ms) - min(s_ms)) / min(s_ms) if s_ms else None
    manifest["environment"]["gpu_state_end"] = gpu_state()
    manifest["finished"] = datetime.now().isoformat()
    manifest["session_seconds"] = round(time.time() - t_session, 1)
    samples_f.close()
    json.dump(correctness, open(os.path.join(out_dir, "correctness.json"), "w"), indent=1)
    json.dump(traces, open(os.path.join(out_dir, "dispatch_trace.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out_dir, "memory.json"), "w"), indent=1)
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    if args.suite != "correctness" and args.rounds > 0:
        analyze([out_dir], out_dir, args.reference)
    print(f"results: {out_dir}  (sentinel drift {manifest['sentinel_drift']})")
    return out_dir


# ---------------------------------------------------------------------------
# Analysis: per-cell summaries and paired ratios with hierarchical bootstrap
# ---------------------------------------------------------------------------

def load_runs(run_dirs):
    """Samples, correctness, memory and matrix from one or more session dirs
    (old single-session runs are read through the rerender adapter)."""
    samples, correctness, memory, matrix, manifests = [], {}, [], {}, []
    for k, d in enumerate(run_dirs):
        man = json.load(open(os.path.join(d, "manifest.json")))
        manifests.append(man)
        session = man.get("session", k)
        for line in open(os.path.join(d, "samples.jsonl")):
            s = json.loads(line)
            s.setdefault("session", session)
            samples.append(s)
        cor = json.load(open(os.path.join(d, "correctness.json")))
        for cfg, cres in cor.items():
            correctness.setdefault(cfg, {}).setdefault(session, adapt_correctness(cres, man))
        mem_path = os.path.join(d, "memory.json")
        if os.path.exists(mem_path):
            memory += [dict(m, session=session) for m in json.load(open(mem_path))]
        for c in man["matrix"]:
            if "mask" not in c:      # experiment-1 manifest: causal windows only
                c = dict(c, mask="window", kv_repr="shared",
                         families=["shared-kv" if c["Hkv"] < c["Hq"] else "core"])
            matrix.setdefault(c["name"], c)
    return samples, correctness, memory, matrix, manifests


def adapt_correctness(cres, manifest):
    """Old (experiment 1) correctness records to the gated schema: their single
    `pass` flag becomes separate fwd/bwd/LSE gates recomputed from the stored
    metrics, with each provider's declared LSE kind."""
    out = {}
    for name, r in cres.get("providers", {}).items():
        if "caps" in r:
            out[name] = r
            continue
        caps = set()
        pinfo = manifest.get("providers", {}).get(name, {})
        ops = set(pinfo.get("ops", []))
        if "fwd" in ops:
            caps.add("fwd")
        if "bwd" in ops:
            caps.add("bwd")
            if "fwd" not in ops:
                caps.add("fwd_state")
        n = dict(r, caps=sorted(caps))
        if "skipped" in r:
            n["status"] = "unsupported"
            n["reason"] = r["skipped"]
        elif "error" in r:
            n["status"] = "error"
        else:
            n["fwd_ok"] = within("Y", r["Y"]) if "Y" in r else ("bwd" in caps)   # bwd-only: state validated via shared fwd
            lse_kind = "other" if name.startswith("fbgemm") else ("natural" if "lse_max_err" in r else None)
            n["lse_kind"] = lse_kind
            n["lse_ok"] = (r.get("lse_ok") if lse_kind == "natural" else None)
            n["bwd_ok"] = all(within(g, r[g]) for g in GRADS if g in r) if "bwd" in caps else None
            n["full_shape_ok"] = (r.get("spot_rows_Y") is None) or within("Y", r["spot_rows_Y"])
            n["status"] = "pass" if (n["fwd_ok"] and n["lse_ok"] is not False and n["bwd_ok"] is not False) else "fail"
        out[name] = n
    return {"providers": out, "oracle": cres.get("oracle_shape"), "oracle_reduced": cres.get("oracle_reduced")}


def bootstrap_ci(per_session, n_boot=2000, seed=0):
    """per_session: {session: [ratio per round]}. Resample sessions, then rounds
    within each, take the median; return (lo, hi) 2.5/97.5 percentiles."""
    rng = random.Random(seed)
    sessions = list(per_session)
    meds = []
    for _ in range(n_boot):
        pick = [rng.choice(sessions) for _ in sessions]
        vals = []
        for s in pick:
            r = per_session[s]
            vals += [rng.choice(r) for _ in r]
        meds.append(statistics.median(vals))
    meds.sort()
    return meds[int(0.025 * (len(meds) - 1))], meds[int(0.975 * (len(meds) - 1))]


def analyze(run_dirs, out_dir, reference):
    samples, correctness, memory, matrix, manifests = load_runs(run_dirs)
    by = {}
    for s in samples:
        by.setdefault((s["config"], s["provider"], s["op"], s["session"], s["round"]), ([], []))
        by[(s["config"], s["provider"], s["op"], s["session"], s["round"])][0].append(s["event_ms"])
        by[(s["config"], s["provider"], s["op"], s["session"], s["round"])][1].append(s["wall_ms"])
    cells = {}
    for (cfg, prov, op, ses, rnd), (ev, wall) in by.items():
        c = cells.setdefault((cfg, prov, op), {"ev": [], "wall": [], "rounds": {}})
        c["ev"] += ev
        c["wall"] += wall
        c["rounds"][(ses, rnd)] = (statistics.median(ev), statistics.median(wall))
    peak = {}
    for m in memory:
        k = (m["config"], m["provider"], m["op"])
        peak[k] = max(peak.get(k, 0), m["peak_incremental_bytes"])

    def gated(cfg, prov, op, ses):
        rec = correctness.get(cfg, {}).get(ses, {}).get("providers", {}).get(prov)
        return bool(rec) and gate(rec, op)

    def gated_all(cfg, prov, op):
        """Publishable only if the provider passed the op's gate in every session it ran in;
        a failure in one session is never hidden by a pass in another."""
        recs = [c.get("providers", {}).get(prov) for c in correctness.get(cfg, {}).values()]
        recs = [r for r in recs if r is not None]
        return bool(recs) and all(gate(r, op) for r in recs)

    present = sorted({p for _, p, _ in cells})
    if reference not in present:
        fallback = next((p for p in present if p.startswith("cuda")), None)
        print(f"reference {reference} absent from these runs; using {fallback}")
        reference = fallback

    def ref_provider(cfg, op):
        if (cfg, reference, op) in cells:
            return reference
        return None

    rows, paired = [], []
    for (cfg, prov, op), c in sorted(cells.items()):
        e, w = summarize(c["ev"]), summarize(c["wall"])
        sessions = sorted({s for s, _ in c["rounds"]})
        all_pass = gated_all(cfg, prov, op)
        row = {"config": cfg, "provider": prov, "op": op, "gate_pass": all_pass,
               "event_median_ms": e["median"], "event_p25_ms": e["p25"], "event_p75_ms": e["p75"], "event_min_ms": e["min"],
               "wall_median_ms": w["median"], "wall_p25_ms": w["p25"], "wall_p75_ms": w["p75"], "wall_min_ms": w["min"],
               "n_samples": e["n"], "n_sessions": len(sessions), "n_rounds": len(c["rounds"]),
               "peak_incremental_bytes": peak.get((cfg, prov, op)),
               "useful_tflops": None, "reference": None,
               "event_speedup_ref_vs_provider": None, "event_speedup_ci95_lo": None, "event_speedup_ci95_hi": None,
               "wall_speedup_ref_vs_provider": None, "wall_speedup_ci95_lo": None, "wall_speedup_ci95_hi": None}
        mcfg = matrix.get(cfg)
        if mcfg and op in ("fwd", "api_fwd") and all_pass:
            fl = useful_flops(mcfg)
            row["useful_tflops"] = fl / (e["median"] * 1e-3) / 1e12
        ref = ref_provider(cfg, op)
        if ref and prov != ref:
            rc = cells[(cfg, ref, op)]
            per_session_ev, per_session_wall = {}, {}
            # speedup = provider_ms / reference_ms: how many times faster the reference is
            # than this provider (> 1: reference faster). With the candidate as reference
            # and a baseline as provider this is baseline_ms / candidate_ms. Event and wall
            # speedups are paired from their own times; API ops are judged on wall time.
            if all_pass and gated_all(cfg, ref, op):
                for (ses, rnd), (pe, pw) in c["rounds"].items():
                    if (ses, rnd) in rc["rounds"]:
                        re_, rw = rc["rounds"][(ses, rnd)]
                        per_session_ev.setdefault(ses, []).append(pe / re_)
                        per_session_wall.setdefault(ses, []).append(pw / rw)
            if per_session_ev:
                allv = [x for v in per_session_ev.values() for x in v]
                allw = [x for v in per_session_wall.values() for x in v]
                lo, hi = bootstrap_ci(per_session_ev)
                wlo, whi = bootstrap_ci(per_session_wall)
                row.update({"reference": ref,
                            "event_speedup_ref_vs_provider": statistics.median(allv), "event_speedup_ci95_lo": lo, "event_speedup_ci95_hi": hi,
                            "wall_speedup_ref_vs_provider": statistics.median(allw), "wall_speedup_ci95_lo": wlo, "wall_speedup_ci95_hi": whi})
                primary = "wall" if op.startswith("api") else "event"
                plo, phi = (wlo, whi) if primary == "wall" else (lo, hi)
                paired.append({"config": cfg, "op": op, "provider": prov, "reference": ref,
                               "n_pairs": len(allv), "n_sessions": len(per_session_ev), "primary_metric": primary,
                               "event_speedup_ref_vs_provider": statistics.median(allv), "event_ci95_lo": lo, "event_ci95_hi": hi,
                               "wall_speedup_ref_vs_provider": statistics.median(allw), "wall_ci95_lo": wlo, "wall_ci95_hi": whi,
                               "ref_faster": plo > 1.0, "provider_faster": phi < 1.0,
                               "provider_median_ms": (w if primary == "wall" else e)["median"],
                               "ref_median_ms": summarize(rc["wall" if primary == "wall" else "ev"])["median"],
                               "families": ",".join(mcfg.get("families", [])) if mcfg else ""})
        rows.append(row)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["config"])
        wri.writeheader()
        wri.writerows(rows)
    with open(os.path.join(out_dir, "paired.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(paired[0].keys()) if paired else ["config"])
        wri.writeheader()
        wri.writerows(paired)
    write_summary_md(out_dir, rows, paired, correctness, matrix, reference, manifests)
    return rows, paired


def family_summary(paired, matrix):
    """Per (family, op, provider): geometric-mean speedup of the reference over the
    provider (primary metric: wall for API ops, events otherwise), counts of cells where
    the reference / the provider is faster with the interval clear of parity, worst and
    best cell from the reference's point of view."""
    out = {}
    for r in paired:
        fams = matrix.get(r["config"], {}).get("families", []) or ["(none)"]
        for fam in fams:
            out.setdefault((fam, r["op"], r["provider"], r["reference"]), []).append(r)
    rows = []
    for (fam, op, prov, ref), rs in sorted(out.items()):
        key = f"{rs[0]['primary_metric']}_speedup_ref_vs_provider"
        vals = [x[key] for x in rs]
        gm = math.exp(sum(math.log(v) for v in vals) / len(vals))
        ref_wins = sum(1 for x in rs if x["ref_faster"])
        prov_wins = sum(1 for x in rs if x["provider_faster"])
        worst = min(rs, key=lambda x: x[key])
        best = max(rs, key=lambda x: x[key])
        rows.append({"family": fam, "op": op, "provider": prov, "reference": ref, "cells": len(rs), "metric": rs[0]["primary_metric"],
                     "geomean_speedup_ref_vs_provider": gm, "ref_faster": ref_wins, "provider_faster": prov_wins,
                     "unresolved": len(rs) - ref_wins - prov_wins,
                     "worst_config": worst["config"], "worst_speedup": worst[key], "best_config": best["config"], "best_speedup": best[key]})
    return rows


def write_summary_md(out_dir, rows, paired, correctness, matrix, reference, manifests):
    provs = sorted({r["provider"] for r in rows})
    lines = [f"# single gather benchmark summary ({os.path.basename(out_dir)})", "",
             f"Reference: `{reference}`. speedup = provider_ms / reference_ms (> 1: the reference is that many times faster "
             "than the provider); device ops pair CUDA-event medians per (session, round), API ops pair wall-clock medians; "
             "the 95% interval is a hierarchical bootstrap over sessions then rounds. A cell without a speedup failed its gate "
             "in some session or the reference did.", ""]
    for m in manifests:
        lines.append(f"- session {m.get('session')}: {m.get('run_id')} started {m.get('started')} "
                     f"sentinel drift {m.get('sentinel_drift')}")
    lines.append("")
    tables = [("fwd", "Forward, prepared inputs, device time (ms, median)"),
              ("bwd", "Backward (full operator incl. KV-gradient reduction), prepared inputs, device time (ms, median)"),
              ("bwd_prepared", "Backward component without the KV-gradient reduction (shared KV only; ms, median)"),
              ("fwd_bwd", "Forward+backward measured together, device time (ms, median)"),
              ("api_fwd", "Forward, API wall time incl. conversions (ms, median)"),
              ("api_fwd_bwd", "Forward+backward, API wall time incl. conversions and grad reduction (ms, median)"),
              ("legacy_state_fwd", "Legacy Q-only three-gather forward used to prepare the ablation's state (ms, median; not a single-gather forward)")]
    for op, title in tables:
        sub = [r for r in rows if r["op"] == op]
        if not sub:
            continue
        cols = [p for p in provs if any(r["provider"] == p for r in sub)]
        lines += [f"## {title}", "", "| config | " + " | ".join(cols) + " | speedup of reference over provider (95% CI) |",
                  "|---|" + "---|" * (len(cols) + 1)]
        for cfg in matrix:
            cells, ratios = [], []
            for p in cols:
                r = next((x for x in sub if x["config"] == cfg and x["provider"] == p), None)
                if r is None:
                    cells.append("-")
                    continue
                key = "wall_median_ms" if op.startswith("api") else "event_median_ms"
                cells.append(f"{r[key]:.3f}" + ("" if r["gate_pass"] else " (GATE FAIL)"))
                k = "wall" if op.startswith("api") else "event"
                if r[f"{k}_speedup_ref_vs_provider"] is not None:
                    ratios.append(f"{p}: {r[f'{k}_speedup_ref_vs_provider']:.2f}x [{r[f'{k}_speedup_ci95_lo']:.2f}, {r[f'{k}_speedup_ci95_hi']:.2f}]")
            if any(x != "-" for x in cells):
                lines.append(f"| {cfg} | " + " | ".join(cells) + " | " + ", ".join(ratios) + " |")
        lines.append("")
    fam = family_summary(paired, matrix)
    if fam:
        lines += ["## Family summary (speedup of the reference over the provider; counts need the interval clear of parity)", "",
                  "| family | op | provider | cells | geomean speedup | ref faster | provider faster | unresolved | worst (config) | best (config) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for f in fam:
            lines.append(f"| {f['family']} | {f['op']} | {f['provider']} | {f['cells']} | {f['geomean_speedup_ref_vs_provider']:.2f}x | "
                         f"{f['ref_faster']} | {f['provider_faster']} | {f['unresolved']} | {f['worst_speedup']:.2f}x ({f['worst_config']}) | {f['best_speedup']:.2f}x ({f['best_config']}) |")
        lines.append("")
        with open(os.path.join(out_dir, "families.csv"), "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(fam[0].keys()))
            wri.writeheader()
            wri.writerows(fam)
    lines += ["## Correctness status (per provider; worst session)", "", "| config | provider | status | Y rel/max | worst grad rel/max | LSE max err | repeat rel L2 | note |", "|---|---|---|---|---|---|---|---|"]
    for cfg in matrix:
        for ses, cres in sorted(correctness.get(cfg, {}).items()):
            for name, r in cres.get("providers", {}).items():
                y = f"{r['Y']['rel_l2']:.4f}/{r['Y']['max_norm']:.4f}" if "Y" in r else "-"
                gs = [r[g] for g in GRADS if g in r]
                g = f"{max(x['rel_l2'] for x in gs):.4f}/{max(x['max_norm'] for x in gs):.4f}" if gs else "-"
                lse = f"{r['lse_max_err']:.4f}" if "lse_max_err" in r else "-"
                rep = f"{r['repeat_max_rel_l2']:.2e}" if "repeat_max_rel_l2" in r else "-"
                note = r.get("reason") or r.get("error") or ""
                lines.append(f"| {cfg} | {name} | {r.get('status')} | {y} | {g} | {lse} | {rep} | {note[:80]} |")
            break
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def rerender(old_dir, reference="cuda_specialized"):
    """Re-render an experiment-1 run from its raw samples under the repaired
    gates and ratio definitions, into a sibling directory; the original files
    are not touched. This is an analysis correction, not a new GPU run."""
    out_dir = old_dir.rstrip("/") + "_rerender"
    os.makedirs(out_dir, exist_ok=True)
    rows, paired = analyze([old_dir], out_dir, reference)
    json.dump({"source": old_dir, "note": "analysis correction: re-rendered from raw samples.jsonl/correctness.json "
               "with per-op gates, explicit event/wall ratios and paired per-round bootstrap intervals; no new GPU run. "
               "Experiment 1 validated no forward state for the bwd-only providers (legacy ablation, Listing 3), so their "
               "fwd_ok is taken from the shared forward they reuse (legacy: Q-only three-gather forward; Listing 3: Listing 1).",
               "rendered": datetime.now().isoformat()}, open(os.path.join(out_dir, "RERENDER.json"), "w"), indent=1)
    print(f"rerendered {old_dir} -> {out_dir} ({len(rows)} cells, {len(paired)} paired ratios)")
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None,
                    help="comma list of families: correctness, core, shared-kv, bh, batch, sharing, mask, seq, "
                         "anchors, holdout, sentinel, all")
    ap.add_argument("--matrix", default=None, help="JSON list of configs instead of --suite")
    ap.add_argument("--providers", default="all", help="comma list of provider names to keep")
    ap.add_argument("--candidates", default=None, help="JSON list of CUDA candidates {name, ext, tiles, window}")
    ap.add_argument("--paper-candidates", default=None, help="JSON list of paper configs {name, fwd, bwd}")
    ap.add_argument("--reference", default="cuda_v0", help="provider the ratios are computed against")
    ap.add_argument("--seed", type=int, default=0, help="input seed (final measurements: 7, 11, 19)")
    ap.add_argument("--seeds", default=None, help="comma list: session k draws inputs with seeds[k %% len]")
    ap.add_argument("--order-seed", type=int, default=0, help="provider-order seed (per session +1000*session)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--tag", default=None, help="run-id prefix (default: suite name)")
    ap.add_argument("--sessions", type=int, default=1, help="independent processes; analysis pools them")
    ap.add_argument("--session", type=int, default=None, help="(internal) session index of this process")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved rounds per session (0: correctness only)")
    ap.add_argument("--iters", type=int, default=0, help="fixed repeats per round (0: calibrate)")
    ap.add_argument("--target-round-ms", type=float, default=150.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--configs", default=None, help="comma list of config names to keep")
    ap.add_argument("--expand-large", action="store_true", help="add B=4/N=512 to the core family")
    ap.add_argument("--fbgemm-path", default=os.environ.get("FBGEMM_SIMPLICIAL", "/home/dev/fbgemm"))
    ap.add_argument("--v0-ext", default=V0_EXT_DEFAULT, help="frozen cf1e258 extension .so for the cuda_v0 provider")
    ap.add_argument("--analyze", nargs="+", default=None, help="run dirs to pool and analyze (no GPU)")
    ap.add_argument("--rerender", default=None, help="experiment-1 run dir to re-render (no GPU)")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.rerender:
        rerender(args.rerender, args.reference if args.reference != "cuda_v0" else "cuda_specialized")
        return
    if args.analyze:
        out = args.output_dir or (args.analyze[0].rstrip("/") + "_pooled")
        analyze(args.analyze, out, args.reference)
        print(f"analysis: {out}")
        return
    if ck is None:
        raise SystemExit(f"att3ntion._cuda_kernels not importable: {_CK_ERR}")
    if not args.suite and not args.matrix:
        raise SystemExit("--suite or --matrix required")
    suites = args.suite.split(",") if args.suite else ["matrix"]
    matrix = load_matrix_file(args.matrix) if args.matrix else build_matrix(suites, args.expand_large)
    if args.configs:
        keep = set(args.configs.split(","))
        matrix = [c for c in matrix if c["name"] in keep]
    tag = args.tag or "-".join(suites)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output_dir or os.path.join(ROOT, "benchmarks", "results", "single_gather", f"{tag}_{stamp}")

    if args.session is None and args.sessions > 1:
        dirs = []
        argv, skip = [], False
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a == "--sessions":
                skip = True
                continue
            if a.startswith("--sessions="):
                continue
            argv.append(a)
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
        for k in range(args.sessions):
            d = f"{base}_s{k}"
            cmd = [sys.executable, os.path.abspath(__file__)] + argv + ["--session", str(k), "--output-dir", d, "--sessions", "1"]
            if seeds:
                cmd += ["--seed", str(seeds[k % len(seeds)])]
            print(f"=== session {k}: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)
            dirs.append(d)
        pooled = f"{base}_pooled"
        analyze(dirs, pooled, args.reference)
        print(f"pooled analysis: {pooled}")
        return

    session = args.session or 0
    out_dir = base if args.output_dir else f"{base}_s{session}"
    os.makedirs(out_dir, exist_ok=True)
    providers = all_providers(args)
    if args.providers != "all":
        keep = args.providers.split(",")
        known = {p.name for p in providers}
        missing = [k for k in keep if k not in known]
        if missing:
            raise SystemExit(f"--providers names not registered: {missing}; registered: {sorted(known)}")
        providers = [p for p in providers if p.name in keep]
    args.suite = ",".join(suites)
    run_session(args, out_dir, matrix, providers, session)


if __name__ == "__main__":
    main()
