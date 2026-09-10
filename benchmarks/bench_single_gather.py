"""Single query-gather comparison: att3ntion CUDA vs the Fast and Simplex paper
kernels and FBGEMM's Triton forward.

    python benchmarks/bench_single_gather.py --suite correctness
    python benchmarks/bench_single_gather.py --suite core
    python benchmarks/bench_single_gather.py --suite shared-kv

Every provider computes the same operator on the same seeded bf16 inputs:
one softmax per query over (j, k) pairs with causal windows w on both key
axes (w = N is full causal), scale 1/sqrt(D), no biases. Correctness is gated
against an fp64 oracle before anything is timed; a provider that fails keeps
its numbers but loses its speedup ratio. Device latency is CUDA events around
the warmed call with output allocation inside; API latency is synchronized
wall clock around the call including layout conversion and KV replication.
"""
import argparse
import csv
import hashlib
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

import att3ntion._cuda_kernels as ck  # noqa: E402

NAMES = ["Q", "R", "S", "Vr", "Vs"]
GRADS = ["dQ", "dR", "dS", "dVr", "dVs"]
TOL = {"Y": (0.03, 0.05), "dVr": (0.04, 0.05), "dVs": (0.04, 0.05),
       "dQ": (0.06, 0.10), "dR": (0.06, 0.10), "dS": (0.06, 0.10)}
LSE_MAX = 0.02
ORACLE_BUDGET = 2 ** 27          # fp64 cube elements the dense reference may hold


# ---------------------------------------------------------------------------
# Inputs and the fp64 oracle
# ---------------------------------------------------------------------------

def window_mask(N, w, B, device):
    i = torch.arange(N, device=device)
    m = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - w)
    return m[None].expand(B, -1, -1).contiguous()


def make_inputs(cfg, seed, device="cuda"):
    """Canonical [B,Hq,N,D] bf16 draws; KV drawn once per KV head."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, Hq, Hkv, N, D = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["N"], cfg["D"]
    q = torch.randn(B, Hq, N, D, generator=g).to(torch.bfloat16).to(device)
    kv = {n: torch.randn(B, Hkv, N, D, generator=g).to(torch.bfloat16).to(device)
          for n in NAMES[1:]}
    dY = torch.randn(B, Hq, N, D, generator=g).to(torch.bfloat16).to(device)
    return {"Q": q, **kv, "dY": dY, "mask": window_mask(N, cfg["w"], B, device), "cfg": cfg}


def expand_kv(t, Hq):
    return t if t.size(1) == Hq else t.expand(-1, Hq, -1, -1).contiguous()


def oracle(inp):
    """fp64 Y, LSE and the five gradients (KV gradients reduced over replicated
    heads when the KV is shared). Holds one [B,Hq,N,N,N] cube at a time."""
    Hq = inp["Q"].size(1)
    leaves = {n: inp[n].double().requires_grad_(True) for n in NAMES}
    Q, R, S, Vr, Vs = (leaves["Q"],) + tuple(expand_kv(leaves[n], Hq) for n in NAMES[1:])
    D = Q.shape[-1]
    x = torch.einsum("bhid,bhjd,bhkd->bhijk", Q, R, S) / math.sqrt(D)
    vis = (inp["mask"][:, :, :, None] & inp["mask"][:, :, None, :])[:, None]
    x = x.masked_fill(~vis, float("-inf")).flatten(3)
    lse = torch.logsumexp(x, -1)
    p = torch.nan_to_num(torch.softmax(x, -1), nan=0.0).reshape(*Q.shape[:3], Q.shape[2], Q.shape[2])
    del x
    Y = torch.einsum("bhijk,bhjd->bhikd", p, Vr)
    Y = torch.einsum("bhikd,bhkd->bhid", Y, Vs)
    Y.backward(inp["dY"].double())
    grads = {g: leaves[n].grad for g, n in zip(GRADS, NAMES)}
    return Y.detach(), lse.detach(), grads


def oracle_rows(inp, rows):
    """fp64 Y for a few query rows at the full shape (no cube)."""
    Hq = inp["Q"].size(1)
    Q = inp["Q"].double()
    R, S, Vr, Vs = (expand_kv(inp[n], Hq).double() for n in NAMES[1:])
    D = Q.shape[-1]
    out = []
    for i in rows:
        x = torch.einsum("bhd,bhjd,bhkd->bhjk", Q[:, :, i], R, S) / math.sqrt(D)
        vis = (inp["mask"][:, i, :, None] & inp["mask"][:, i, None, :])[:, None]
        p = torch.softmax(x.masked_fill(~vis, float("-inf")).flatten(2), -1)
        p = torch.nan_to_num(p, nan=0.0).reshape(x.shape)
        out.append(torch.einsum("bhjk,bhjd,bhkd->bhd", p, Vr, Vs))
    return torch.stack(out, 2)


def err_metrics(actual, ref):
    a, r = actual.double(), ref.double()
    rel = ((a - r).norm() / max(r.norm().item(), 1e-12)).item()
    mx = ((a - r).abs().max() / max(1.0, r.abs().max().item())).item()
    return {"rel_l2": rel, "max_norm": mx, "finite": bool(torch.isfinite(a).all())}


def within(name, m):
    rt, mt = TOL[name]
    return m["finite"] and m["rel_l2"] <= rt and m["max_norm"] <= mt


def useful_flops(cfg):
    N, w = cfg["N"], cfg["w"]
    pairs = sum(min(i + 1, w) ** 2 for i in range(N))
    return 4 * cfg["D"] * cfg["B"] * cfg["Hq"] * pairs


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
# Each provider converts the canonical inputs into its own layout (prepare),
# then exposes fwd / bwd on prepared state. Results come back canonical.

class Provider:
    name = ""
    ops = ("fwd", "bwd", "fwd_bwd")
    layout = ""
    unavailable = None          # reason string when the provider cannot run

    def check(self, cfg):
        return None             # reason string when this config is unsupported

    def prepare(self, inp):
        raise NotImplementedError

    def fwd(self, st):
        raise NotImplementedError

    def bwd(self, st, out):
        raise NotImplementedError

    def to_canonical_Y(self, st, out):
        return out[0]

    def lse(self, st, out):
        return None

    def to_canonical_grads(self, st, grads):
        return dict(zip(GRADS, grads))

    def kv_grad_reduce(self, inp, grads):
        """Shared KV: replicated per-head KV gradients reduce to one head."""
        if inp["cfg"]["Hkv"] == 1 and inp["Q"].size(1) > 1:
            for n in GRADS[1:]:
                grads[n] = grads[n].float().sum(1, keepdim=True).to(grads[n].dtype)
        return grads


class CudaSpecialized(Provider):
    name = "cuda_specialized"
    layout = "[B,H,N,D] bf16 native; shared KV replicated to Hq heads"

    def prepare(self, inp):
        Hq = inp["Q"].size(1)
        st = {n: expand_kv(inp[n], Hq) for n in NAMES}
        st["dY"], st["mask"] = inp["dY"], inp["mask"]
        return st

    def fwd(self, st):
        return ck.single_gather_forward(*[st[n] for n in NAMES], st["mask"])

    def bwd(self, st, out):
        return ck.single_gather_backward(st["dY"], *[st[n] for n in NAMES], *out, st["mask"])

    def lse(self, st, out):
        m, l = out[1], out[2]
        return torch.where(l > 0, m.double() + torch.log(l.double().clamp_min(1e-300)),
                           torch.full_like(m.double(), float("-inf")))


class CudaLegacyUnspecialized(Provider):
    """Ablation: the legacy three-gather backward (Bwd_gather_tc<BWD_ALL> x3)
    fed zero cotangents for the two unused gathers. Its forward is the legacy
    Q-only forward, which also fabricates the absent gathers' stats and zero
    outputs; that state is prepared outside the backward timing and its cost
    is reported as `legacy_state_fwd`, not as a single-gather forward."""
    name = "cuda_unspecialized_legacy"
    ops = ("bwd", "legacy_state_fwd")
    layout = "[B,H,N,D] bf16; 9-input legacy API with zero Vq/V2"

    def prepare(self, inp):
        Hq = inp["Q"].size(1)
        st = {n: expand_kv(inp[n], Hq) for n in NAMES}
        st["zero"] = torch.zeros_like(st["Q"])
        st["dY"], st["mask"] = inp["dY"], inp["mask"]
        st["N"] = st["Q"].size(2)
        return st

    def fwd(self, st):
        z, N = st["zero"], st["N"]
        return ck.forward(st["Q"], st["R"], st["S"], z, z, st["Vr"], z, st["Vs"], z,
                          0.0, N, N, N, st["mask"], 1)

    def bwd(self, st, out):
        z = st["zero"]
        g = ck.backward(st["dY"], z, z, z, z, z, st["Q"], st["R"], st["S"], z, z,
                        st["Vr"], z, st["Vs"], z, *out[6:12], 0.0, st["mask"],
                        out[0], out[1], out[2])
        return g[0], g[1], g[2], g[5], g[7]

    def lse(self, st, out):
        m, l = out[6], out[7]
        return torch.where(l > 0, m.double() + torch.log(l.double().clamp_min(1e-300)),
                           torch.full_like(m.double(), float("-inf")))


class PaperTriton(Provider):
    """Appendix listings via benchmarks/paper_simplicial.py. Layout [b,s,k,h]."""
    name = "paper_triton"
    layout = "[b,s,k,h] bf16 (permute of canonical); shared KV replicated to Hq heads"
    bwd_kind = "general"

    def __init__(self):
        try:
            import paper_simplicial
            self.ps = paper_simplicial
        except Exception as e:      # noqa: BLE001
            self.unavailable = f"paper_simplicial import failed: {e!r}"

    def check(self, cfg):
        return None

    def prepare(self, inp):
        Hq = inp["Q"].size(1)
        to = lambda t: t.permute(0, 2, 1, 3).contiguous()
        st = {n: to(expand_kv(inp[n], Hq)) for n in NAMES}
        st["dY"] = to(inp["dY"])
        st["w"] = inp["cfg"]["w"]
        return st

    def fwd(self, st):
        return self.ps.paper_fwd(*[st[n] for n in NAMES], st["w"], st["w"])

    def bwd(self, st, out):
        fn = self.ps.paper_bwd_general if self.bwd_kind == "general" else self.ps.paper_bwd_small_w2
        return fn(*[st[n] for n in NAMES], out[0], out[1], st["dY"], st["w"], st["w"])

    def to_canonical_Y(self, st, out):
        return out[0].permute(0, 2, 1, 3).contiguous()

    def lse(self, st, out):
        return out[1].double()

    def to_canonical_grads(self, st, grads):
        return {g: t.permute(0, 2, 1, 3).contiguous() for g, t in zip(GRADS, grads)}


class PaperTritonSmallW2(PaperTriton):
    """Listing 3 (no atomics) for dQ/dK2/dV2 + Listing 2 for dK1/dV1; w must be 32."""
    name = "paper_triton_listing3_w32"
    ops = ("bwd", "fwd_bwd")
    bwd_kind = "small_w2"

    def check(self, cfg):
        if cfg["w"] != 32:
            return "Listing 3 needs w2 = BLOCK_SIZE_KV2 - BLOCK_SIZE_Q = 32"
        return None


class FbgemmTritonFwd(Provider):
    """FBGEMM simplicial.ops.triton.fwd GQA-packed forward (kv heads = 1, all
    query heads form the M tile). Launched directly so no bias path is touched:
    the upstream `triton_fwd` wrapper's `if not k2_bias` would re-enable
    bias = 1/head_dim for a zero argument. Forward only: its backward is not
    validated here and its saved statistic is not a natural-log LSE."""
    name = "fbgemm_triton_fwd"
    ops = ("fwd",)
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
        if cfg["Hkv"] != 1:
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


class TlxFwd(Provider):
    name = "tlx_fwd_ws"
    ops = ("fwd",)

    def __init__(self, fbgemm_path):
        if fbgemm_path and fbgemm_path not in sys.path:
            sys.path.insert(0, fbgemm_path)
        try:
            import triton.language.extra.tlx  # noqa: F401
            from simplicial.ops.tlx import fwd_ws  # noqa: F401
        except Exception as e:      # noqa: BLE001
            self.unavailable = f"TLX not importable ({e!r}); needs the TLX Triton fork"
            return
        self.unavailable = "TLX provider not wired in this shift"


def all_providers(fbgemm_path):
    return [CudaSpecialized(), CudaLegacyUnspecialized(), PaperTriton(),
            PaperTritonSmallW2(), FbgemmTritonFwd(fbgemm_path), TlxFwd(fbgemm_path)]


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def core_matrix(expand_large):
    cfgs = [dict(B=1, Hq=H, Hkv=H, N=N, D=D, w=w)
            for H in (1, 4) for D in (64, 128) for N in (128, 256) for w in (N, 32)]
    if expand_large:
        cfgs += [dict(B=4, Hq=H, Hkv=H, N=512, D=D, w=w)
                 for H in (1, 4) for D in (64, 128) for w in (512, 32)]
    return [dict(c, name=f"B{c['B']}_H{c['Hq']}_D{c['D']}_N{c['N']}_w{'full' if c['w'] == c['N'] else c['w']}")
            for c in cfgs]


def shared_kv_matrix():
    cfgs = [dict(B=1, Hq=64, Hkv=1, N=N, D=128, w=w) for N in (128, 256, 512) for w in (N, 32)]
    return [dict(c, name=f"sharedkv_Hq64_D128_N{c['N']}_w{'full' if c['w'] == c['N'] else c['w']}")
            for c in cfgs]


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------

def oracle_shape(cfg):
    B, H = cfg["B"], cfg["Hq"]
    while B * H * cfg["N"] ** 3 > ORACLE_BUDGET and (H > 1 or B > 1):
        if H > 1:
            H = max(1, H // 2)
        else:
            B = max(1, B // 2)
    return dict(cfg, B=B, Hq=H, Hkv=min(cfg["Hkv"], H))


def slice_inputs(inp, sub):
    out = {"cfg": sub, "mask": inp["mask"][:sub["B"]].contiguous()}
    for n in NAMES + ["dY"]:
        t = inp[n]
        out[n] = t[:sub["B"], :min(t.size(1), sub["Hq"] if n in ("Q", "dY") else sub["Hkv"])].contiguous()
    return out


def check_provider(p, inp, ref, ref_rows, rows):
    """Run p on the oracle-shaped inputs and compare; also spot-check Y rows at
    the full shape. Returns a dict with pass flags and metrics."""
    Y_ref, lse_ref, g_ref = ref
    res = {"provider": p.name, "checked_ops": []}
    st = p.prepare(inp)
    out = p.fwd(st)
    torch.cuda.synchronize()
    ok = True
    if "fwd" in p.ops:
        Y = p.to_canonical_Y(st, out)
        res["Y"] = err_metrics(Y, Y_ref)
        ok &= within("Y", res["Y"])
        lse = p.lse(st, out)
        if lse is not None:
            dead = lse_ref == float("-inf")
            live = ~dead
            res["lse_dead_rows_match"] = bool(torch.equal(dead, lse == float("-inf")))
            res["lse_max_err"] = (lse[live] - lse_ref[live]).abs().max().item() if live.any() else 0.0
            res["lse_ok"] = res["lse_dead_rows_match"] and res["lse_max_err"] <= LSE_MAX
        res["checked_ops"].append("fwd")
    if "bwd" in p.ops:
        grads = p.kv_grad_reduce(inp, p.to_canonical_grads(st, p.bwd(st, out)))
        torch.cuda.synchronize()
        for g in GRADS:
            res[g] = err_metrics(grads[g], g_ref[g])
            ok &= within(g, res[g])
        res["checked_ops"].append("bwd")
    res["pass"] = bool(ok)
    return res, st, out


def spot_check_rows(p, st, out, ref_rows, rows):
    if "fwd" not in p.ops:
        return None
    Y = p.to_canonical_Y(st, out)[:, :, rows]
    return err_metrics(Y, ref_rows)


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
    return ev, wall, torch.cuda.max_memory_allocated(), base


def op_callable(p, op, inp, st, out):
    if op == "fwd":
        return lambda: p.fwd(st)
    if op == "bwd":
        return lambda: p.bwd(st, out)
    if op == "fwd_bwd":
        return lambda: p.bwd(st, p.fwd(st))
    if op == "legacy_state_fwd":
        return lambda: p.fwd(st)
    if op == "api_fwd":
        return lambda: p.to_canonical_Y(*(lambda s: (s, p.fwd(s)))(p.prepare(inp)))
    if op == "api_fwd_bwd":
        def run():
            s = p.prepare(inp)
            o = p.fwd(s)
            return p.kv_grad_reduce(inp, p.to_canonical_grads(s, p.bwd(s, o)))
        return run
    raise ValueError(op)


def summarize(xs):
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"median": statistics.median(xs), "p25": q(0.25), "p75": q(0.75), "min": xs[0], "n": len(xs)}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def environment(fbgemm_path):
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    files = ["cuda/forward.cu", "cuda/backward.cu", "cuda/common.cuh", "cpp/cuda_bindings.cpp",
             "cpp/cuda_bindings.h", "att3ntion/_single_gather.py", "benchmarks/bench_single_gather.py",
             "benchmarks/paper_simplicial.py", "tests/test_single_gather.py"]
    hashes = {f: sha256_file(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))}
    fb = os.path.join(fbgemm_path or "", "simplicial/ops/triton/fwd.py")
    if os.path.exists(fb):
        hashes["fbgemm/simplicial/ops/triton/fwd.py"] = sha256_file(fb)
    commit = os.path.join(fbgemm_path or "", "SOURCE_COMMIT.txt")
    try:
        import triton
        triton_v = triton.__version__
    except Exception:       # noqa: BLE001
        triton_v = None
    return {
        "gpu": props.name, "sm_count": props.multi_processor_count,
        "smem_optin": props.shared_memory_per_block_optin, "nvidia_smi": smi,
        "competing_processes": procs or "none", "torch": torch.__version__,
        "cuda": torch.version.cuda, "triton": triton_v, "python": platform.python_version(),
        "extension": ck.__file__, "source_sha256": hashes,
        "fbgemm_commit": open(commit).read().strip() if os.path.exists(commit) else None,
        "hostname": platform.node(),
    }


def dispatch_trace(p, inp):
    from torch.profiler import ProfilerActivity, profile
    st = p.prepare(inp)
    out = p.fwd(st)
    if "bwd" in p.ops:
        p.bwd(st, out)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = p.fwd(st)
        if "bwd" in p.ops:
            p.bwd(st, out)
        torch.cuda.synchronize()
    kernels = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0) > 0:
            kernels[e.key] = {"count": e.count, "device_us": round(getattr(e, "self_device_time_total", 0.0), 1)}
    return {k: v for k, v in kernels.items() if v["count"] > 0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["correctness", "core", "shared-kv"], required=True)
    ap.add_argument("--providers", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--configs", default=None, help="comma list of config names to keep")
    ap.add_argument("--expand-large", action="store_true", help="add B=4/N=512 to the core matrix")
    ap.add_argument("--fbgemm-path", default=os.environ.get("FBGEMM_SIMPLICIAL", "/home/dev/fbgemm"))
    args = ap.parse_args()

    run_id = datetime.now().strftime(f"{args.suite}_%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(ROOT, "benchmarks", "results", "single_gather", run_id)
    os.makedirs(out_dir, exist_ok=True)

    providers = all_providers(args.fbgemm_path)
    if args.providers != "all":
        keep = set(args.providers.split(","))
        providers = [p for p in providers if p.name in keep]
    if args.suite == "correctness":
        matrix = core_matrix(args.expand_large) + shared_kv_matrix()
    elif args.suite == "core":
        matrix = core_matrix(args.expand_large)
    else:
        matrix = shared_kv_matrix()
    if args.configs:
        keep = set(args.configs.split(","))
        matrix = [c for c in matrix if c["name"] in keep]

    rng = random.Random(args.seed)
    manifest = {
        "run_id": run_id, "suite": args.suite, "command": " ".join(sys.argv), "args": vars(args),
        "environment": environment(args.fbgemm_path), "started": datetime.now().isoformat(),
        "providers": {p.name: {"ops": list(p.ops), "layout": p.layout, "unavailable": p.unavailable}
                      for p in all_providers(args.fbgemm_path)},
        "paper_config": {"forward": "BLOCK_SIZE_Q=32 BLOCK_SIZE_KV=64 num_warps=4 num_stages=1",
                         "backward_general": "BLOCK_SIZE_Q=32 BLOCK_SIZE_KV=64 num_warps=4 num_stages=1",
                         "backward_listing3": "BLOCK_SIZE_Q=32 BLOCK_SIZE_KV2=64 num_warps=4 num_stages=1"},
        "matrix": matrix, "rounds": [], "notes": [
            "windows: a query i sees keys j with i-w < j <= i on both axes; w=N is full causal",
            "device latency: CUDA events around the warmed call, output allocation inside the call",
            "api latency: synchronized wall clock incl. layout conversion / KV replication / grad reduction",
            "cuda_unspecialized_legacy: backward only; legacy_state_fwd is its Q-only three-gather forward, not a single-gather forward",
            "fbgemm_triton_fwd: forward only; saved statistic is log2-domain max + ln(l), not natural-log LSE (reported as lse_max_err)",
            "local CUDA kernel traverses dense (j,k) pairs under the window mask; no window skipping",
        ],
    }
    print(f"[{run_id}] gpu={manifest['environment']['gpu']} ext={ck.__file__}")
    for p in providers:
        if p.unavailable:
            print(f"  provider {p.name}: UNAVAILABLE ({p.unavailable})")

    correctness, traces, samples, summary = {}, {}, [], []
    samples_f = open(os.path.join(out_dir, "samples.jsonl"), "w")

    for cfg in matrix:
        inp = make_inputs(cfg, args.seed)
        sub = oracle_shape(cfg)
        sub_inp = inp if sub["B"] == cfg["B"] and sub["Hq"] == cfg["Hq"] else slice_inputs(inp, sub)
        ref = oracle(sub_inp)
        rows = sorted({0, cfg["N"] // 3, 2 * cfg["N"] // 3, cfg["N"] - 1})
        ref_rows = oracle_rows(inp, rows)
        cres = {"oracle_shape": {k: sub[k] for k in ("B", "Hq", "Hkv", "N", "D", "w")},
                "oracle_reduced": sub_inp is not inp, "spot_rows": rows, "providers": {}}
        active = []
        full_grads = {}
        for p in providers:
            reason = p.unavailable or p.check(cfg)
            if reason:
                cres["providers"][p.name] = {"skipped": reason}
                continue
            try:
                r, _, _ = check_provider(p, sub_inp, ref, ref_rows, rows)
                st, out = p.prepare(inp), None
                out = p.fwd(st)
                r["spot_rows_Y"] = spot_check_rows(p, st, out, ref_rows, rows)
                if "bwd" in p.ops:
                    full_grads[p.name] = p.kv_grad_reduce(inp, p.to_canonical_grads(st, p.bwd(st, out)))
                torch.cuda.synchronize()
                if p.name == "cuda_specialized":
                    r["dispatch"] = ck.sg_dispatch()
                traces.setdefault(cfg["name"], {})[p.name] = dispatch_trace(p, inp)
                cres["providers"][p.name] = r
                active.append((p, st, out))
                flag = "PASS" if r["pass"] else "FAIL"
                print(f"  {cfg['name']:32s} {p.name:28s} {flag}  " + " ".join(
                    f"{k}={r[k]['rel_l2']:.4f}/{r[k]['max_norm']:.4f}" for k in ["Y"] + GRADS if k in r))
            except Exception as e:      # noqa: BLE001
                cres["providers"][p.name] = {"error": repr(e), "pass": False}
                print(f"  {cfg['name']:32s} {p.name:28s} ERROR {e!r}")
        if "cuda_specialized" in full_grads:
            for name, g in full_grads.items():
                if name != "cuda_specialized":
                    cres["providers"][name]["vs_cuda_specialized_full_shape"] = {
                        k: err_metrics(g[k], full_grads["cuda_specialized"][k]) for k in GRADS}
        correctness[cfg["name"]] = cres
        if args.suite == "correctness":
            continue

        order = [p.name for p, _, _ in active]
        for rnd in range(args.rounds):
            rng.shuffle(order)
            manifest["rounds"].append({"config": cfg["name"], "round": rnd, "order": list(order)})
            for pname in order:
                p, st, out = next(t for t in active if t[0].name == pname)
                ops = list(p.ops) + (["api_fwd"] if "fwd" in p.ops else []) \
                    + (["api_fwd_bwd"] if "fwd_bwd" in p.ops else [])
                for op in ops:
                    fn = op_callable(p, op, inp, st, out)
                    ev, wall, peak, base = time_op(fn, args.warmup if rnd == 0 else 2, args.iters)
                    for i, (a, b) in enumerate(zip(ev, wall)):
                        rec = {"config": cfg["name"], "provider": pname, "op": op, "round": rnd,
                               "iter": i, "event_ms": a, "wall_ms": b}
                        samples.append(rec)
                        samples_f.write(json.dumps(rec) + "\n")
                    summary.append({"config": cfg["name"], "provider": pname, "op": op, "round": rnd,
                                    "peak_alloc_bytes": peak, "alloc_before_bytes": base})
        samples_f.flush()
        torch.cuda.empty_cache()

    samples_f.close()
    json.dump(correctness, open(os.path.join(out_dir, "correctness.json"), "w"), indent=1)
    json.dump(traces, open(os.path.join(out_dir, "dispatch_trace.json"), "w"), indent=1)
    manifest["finished"] = datetime.now().isoformat()

    if args.suite != "correctness":
        write_summary(out_dir, matrix, samples, summary, correctness, providers)
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"results: {out_dir}")


def write_summary(out_dir, matrix, samples, peaks, correctness, providers):
    by = {}
    for s in samples:
        by.setdefault((s["config"], s["provider"], s["op"]), ([], []))
        by[(s["config"], s["provider"], s["op"])][0].append(s["event_ms"])
        by[(s["config"], s["provider"], s["op"])][1].append(s["wall_ms"])
    peak = {}
    for r in peaks:
        k = (r["config"], r["provider"], r["op"])
        peak[k] = max(peak.get(k, 0), r["peak_alloc_bytes"])
    rows = []
    for (cfg, prov, op), (ev, wall) in by.items():
        c = next(m for m in matrix if m["name"] == cfg)
        cr = correctness[cfg]["providers"].get(prov, {})
        passed = cr.get("pass", False)
        e, w = summarize(ev), summarize(wall)
        cuda_key = (cfg, "cuda_specialized", op if op != "legacy_state_fwd" else "fwd")
        ratio = None
        if prov != "cuda_specialized" and passed and cuda_key in by and op != "legacy_state_fwd":
            ratio = statistics.median(by[cuda_key][0]) / e["median"]
        flops = useful_flops(c) if op in ("fwd", "api_fwd") else None
        rows.append({"config": cfg, "provider": prov, "op": op, "correct": passed,
                     "event_median_ms": e["median"], "event_p25_ms": e["p25"], "event_p75_ms": e["p75"],
                     "event_min_ms": e["min"], "wall_median_ms": w["median"], "wall_p25_ms": w["p25"],
                     "wall_p75_ms": w["p75"], "wall_min_ms": w["min"], "n": e["n"],
                     "peak_alloc_bytes": peak.get((cfg, prov, op)),
                     "speedup_cuda_over_provider": ratio,
                     "useful_tflops": (flops / (e["median"] * 1e-3) / 1e12) if flops else None,
                     "Y_rel_l2": cr.get("Y", {}).get("rel_l2"), "Y_max_norm": cr.get("Y", {}).get("max_norm"),
                     **{f"{g}_rel_l2": cr.get(g, {}).get("rel_l2") for g in GRADS}})
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    provs = [p.name for p in providers if not p.unavailable]
    lines = [f"# single gather benchmark summary ({os.path.basename(out_dir)})", ""]
    tables = [("fwd", "Forward, prepared inputs, device time (ms, median)"),
              ("bwd", "Backward, prepared inputs, device time (ms, median)"),
              ("fwd_bwd", "Forward+backward measured together, device time (ms, median)"),
              ("api_fwd", "Forward, API wall time incl. conversions (ms, median)"),
              ("api_fwd_bwd", "Forward+backward, API wall time incl. conversions and grad reduction (ms, median)"),
              ("legacy_state_fwd", "Legacy Q-only three-gather forward used to prepare the ablation's state (ms, median; not a single-gather forward)")]
    for op, title in tables:
        sub = [r for r in rows if r["op"] == op]
        if not sub:
            continue
        cols = [p for p in provs if any(r["provider"] == p for r in sub)]
        lines += [f"## {title}", "", "| config | " + " | ".join(cols) + " | ratio (provider_ms / cuda_ms) |",
                  "|---|" + "---|" * (len(cols) + 1)]
        for c in matrix:
            cells, ratios = [], []
            for p in cols:
                r = next((x for x in sub if x["config"] == c["name"] and x["provider"] == p), None)
                if r is None:
                    cells.append("-")
                    continue
                key = "wall_median_ms" if op.startswith("api") else "event_median_ms"
                cells.append(f"{r[key]:.3f}" + ("" if r["correct"] else " (FAILED)"))
                if r["speedup_cuda_over_provider"] is not None:
                    ratios.append(f"{p}: {1.0 / r['speedup_cuda_over_provider']:.2f}x")
            if any(x != "-" for x in cells):
                lines.append(f"| {c['name']} | " + " | ".join(cells) + " | " + ", ".join(ratios) + " |")
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
