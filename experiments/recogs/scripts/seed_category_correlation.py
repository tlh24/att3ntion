"""Across-seed correlation of per-category gen EM.

Colleague's hypothesis: seed sensitivity reflects whether SGD found the right
structural solution -> per-category performance should be POSITIVELY correlated
across replicates (a 'good seed' does well everywhere, esp. structural cats).
Null: seeds vary independently per category -> correlations ~0.
"""
import json, glob, sys
import numpy as np

BASE = "/home/anosha/att3ntion/experiments/recogs/runs"
STRUCTURAL = {"obj_pp_to_subj_pp", "cp_recursion", "pp_recursion"}

ARMS = {
    "v3 hg_gather (10s)":  f"{BASE}/v3/hg_gather_hypergraph_cuda_s*",
    "v3 graph_flash (10s)": f"{BASE}/v3/graph_flash_graph_flash_s*",
    "v8 graph_d870 (8s)": f"{BASE}/v8/graph_d870_300ep_graph_flash_s*",
    "v8 hg_d768 (4s)": f"{BASE}/v8/hg_d768_300ep_hypergraph_cuda_s*",
    "v5 hg_d768 (10s)": f"{BASE}/v5/hg_d768_hypergraph_cuda_s*",
    "v5 hg_d512": f"{BASE}/v5/hg_d512_hypergraph_cuda_s*",
    "v6 hg_d256_l2 (8s)": f"{BASE}/v6/hg_d256_l2_hypergraph_cuda_s*",
}

def load(globpat):
    rows = {}
    for d in sorted(glob.glob(globpat)):
        try:
            m = json.load(open(f"{d}/final_metrics.json"))
        except FileNotFoundError:
            continue
        seed = d.rsplit("_s", 1)[1]
        rows[int(seed)] = {c: v["exact_match"] for c, v in m["gen"]["per_category"].items()}
    return rows

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return np.corrcoef(ra, rb)[0, 1]

for arm, pat in ARMS.items():
    rows = load(pat)
    if len(rows) < 4:
        print(f"\n=== {arm}: only {len(rows)} seeds with final_metrics, skipping"); continue
    seeds = sorted(rows)
    cats = sorted(rows[seeds[0]])
    M = np.array([[rows[s][c] for c in cats] for s in seeds])  # seeds x cats
    keep = [j for j in range(len(cats)) if M[:, j].std() > 1e-9]
    dropped = [cats[j] for j in range(len(cats)) if j not in keep]
    print(f"\n=== {arm} | {len(seeds)} seeds | {len(keep)}/{len(cats)} cats with seed variance")
    if dropped:
        print(f"    no-variance (dropped): {', '.join(dropped)}")

    # 1) each category vs leave-one-out mean of the rest
    print(f"    {'category':52s} {'mean':>6s} {'std':>6s}  rho(cat, rest)")
    loo = []
    for j in keep:
        rest = M[:, [k for k in keep if k != j]].mean(axis=1)
        r = spearman(M[:, j], rest)
        loo.append(r)
        tag = " STRUCT" if cats[j] in STRUCTURAL else ""
        print(f"    {cats[j]:52s} {M[:,j].mean():6.3f} {M[:,j].std():6.3f}  {r:+.2f}{tag}")
    print(f"    median rho(cat, rest-of-gen) = {np.nanmedian(loo):+.2f}")

    # 2) pairwise among structural cats
    sj = [j for j in keep if cats[j] in STRUCTURAL]
    for a in range(len(sj)):
        for b in range(a + 1, len(sj)):
            r = spearman(M[:, sj[a]], M[:, sj[b]])
            print(f"    struct pair {cats[sj[a]]} ~ {cats[sj[b]]}: rho={r:+.2f}")

    # 3) one-factor check: PC1 variance share of standardized matrix
    Z = (M[:, keep] - M[:, keep].mean(0)) / M[:, keep].std(0)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    print(f"    PC1 variance share = {ev[0]/ev.sum():.0%} (1/k noise floor ≈ {1/len(keep):.0%})")

    # 4) overall gen EM spread across seeds
    overall = M[:, keep].mean(axis=1)
    print(f"    per-seed mean-over-cats EM: min={overall.min():.3f} max={overall.max():.3f}")
