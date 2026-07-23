#!/usr/bin/env python3
"""Per-category ReCOGS generalization chart, RUN_4 (v4): graph (FLOP-matched,
6L/d1088, exp/v4_flop_match/runs) vs hypergraph (hg_gather 3L, exp/v3_prefix_lm/runs), WITH an in-distribution
test-set reference row on the same axis.

Identical layout to plots/plot_gen_with_test_v3.py — the only change is that the
two arms are sourced from different run dirs (graph from v4, hg from v3) since the
FLOP-matched graph is the v4 arm while hg is reused from v3.
"""
import glob, json, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

# Each arm sourced from its own runs dir. Blue = graph, red = hypergraph (as v3).
ARM_GLOB = {
    "hypergraph": "exp/v3_prefix_lm/runs/hg_gather_*",
    "graph (FLOP-matched)": "exp/v4_flop_match/runs/graph_flop6L_*",
}


def gen_dirs(arm):
    """Run dirs (per seed) that have a per_example_gen.json, sorted."""
    return sorted(
        os.path.dirname(p)
        for p in glob.glob(os.path.join(ROOT, ARM_GLOB[arm], "per_example_gen.json"))
    )


def gen_per_category(dirs):
    """{category: [acc% per seed]} for the gen split."""
    per_seed = []
    for d in dirs:
        rows = json.load(open(os.path.join(d, "per_example_gen.json")))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            agg[r["category"]][0] += r["correct"]
            agg[r["category"]][1] += 1
        per_seed.append({c: v[0] / v[1] * 100 for c, v in agg.items()})
    cats = set().union(*[s.keys() for s in per_seed]) if per_seed else set()
    return {c: [s[c] for s in per_seed if c in s] for c in cats}


def test_overall(dirs):
    """[overall test EM% per seed] for the same run dirs (skip dirs lacking the file)."""
    out = []
    for d in dirs:
        p = os.path.join(d, "per_example_test.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        if rows:
            out.append(100 * sum(r["correct"] for r in rows) / len(rows))
    return out


LABELS = {
    "subj_to_obj_common": "common noun: subject → object",
    "subj_to_obj_proper": "proper noun: subject → object",
    "obj_to_subj_common": "common noun: object → subject",
    "obj_to_subj_proper": "proper noun: object → subject",
    "prim_to_subj_common": "common noun: isolated → subject",
    "prim_to_subj_proper": "proper noun: isolated → subject",
    "prim_to_obj_common": "common noun: isolated → object",
    "prim_to_obj_proper": "proper noun: isolated → object",
    "prim_to_inf_arg": "verb: isolated → infinitive argument",
    "active_to_passive": "verb: active → passive",
    "passive_to_active": "verb: passive → active",
    "obj_omitted_transitive_to_transitive": "verb: object-omitted → transitive (w/ object)",
    "unacc_to_transitive": "verb: unaccusative → transitive",
    "do_dative_to_pp_dative": "dative: double-object → prepositional",
    "pp_dative_to_do_dative": "dative: prepositional → double-object",
    "only_seen_as_transitive_subj_as_unacc_subj": "subject: transitive-only → unaccusative",
    "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj": "subject: unaccusative-only → obj-omitted transitive",
    "only_seen_as_unacc_subj_as_unerg_subj": "subject: unaccusative-only → unergative",
    "obj_pp_to_subj_pp": "PP modifier: on object → on subject",
    "cp_recursion": "structural: deeper CP (clause) recursion",
    "pp_recursion": "structural: deeper PP (prep-phrase) recursion",
}
label = lambda c: LABELS.get(c, c)
STRUCTURAL = {"obj_pp_to_subj_pp", "pp_recursion", "cp_recursion"}

gf_dirs, hg_dirs = gen_dirs("graph (FLOP-matched)"), gen_dirs("hypergraph")
gf, hg = gen_per_category(gf_dirs), gen_per_category(hg_dirs)
gf_test, hg_test = test_overall(gf_dirs), test_overall(hg_dirs)

all_cats = set(gf) | set(hg)
_by_gf = lambda c: np.mean(gf.get(c, [0]))
lexical = sorted(all_cats - STRUCTURAL, key=_by_gf, reverse=True)
structural = sorted(all_cats & STRUCTURAL, key=_by_gf, reverse=True)


def stat(d, c):
    v = d.get(c, [0])
    return np.mean(v), np.std(v)


sections = [
    ("IN-DISTRIBUTION (test)", "#000", ["__test__"]),
    ("lexical generalization", "#555", lexical),
    ("structural generalization", "#555", structural),
]

rows, headers = [], []
y, cur, GAP = [], 0.0, 1.4
ood_label_y = None
for title, color, keys in sections:
    if not keys:
        continue
    if title.startswith("lexical") and ood_label_y is None:
        ood_label_y = cur - 1.0
        cur += 1.0
    headers.append((cur - 1.0, title, color))
    for k in keys:
        if k == "__test__":
            rows.append(("in-distribution (test set)",
                         np.mean(gf_test), np.std(gf_test), np.mean(hg_test), np.std(hg_test)))
        else:
            rows.append((label(k), *stat(gf, k), *stat(hg, k)))
        y.append(cur)
        cur += 1.0
    cur += GAP
y = np.array(y)

gf_mean = [r[1] for r in rows]; gf_err = [r[2] for r in rows]
hg_mean = [r[3] for r in rows]; hg_err = [r[4] for r in rows]

h = 0.4
fig, ax = plt.subplots(figsize=(13, 10))

n_test = len(sections[0][2])
n_lex = len(lexical)
if n_lex:
    ax.axhspan(y[n_test] - 0.5, y[n_test + n_lex - 1] + 0.5, color="#bfe3bf", alpha=0.8, zorder=0)

ax.barh(y + h / 2, gf_mean, h, xerr=gf_err, label="graph (FLOP-matched)",
        color="#1f77b4", error_kw=dict(ecolor="#444", lw=0.8, capsize=2), zorder=3)
ax.barh(y - h / 2, hg_mean, h, xerr=hg_err, label="hypergraph",
        color="#d62728", error_kw=dict(ecolor="#444", lw=0.8, capsize=2), zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("exact-match accuracy (%)")
ax.set_xlim(0, 100)
ax.set_ylim(y[-1] + 1.0, y[0] - 1.4)

for hy, text, color in headers:
    ax.text(99, hy, text, ha="right", va="center", fontsize=10, fontweight="bold", color=color)

if ood_label_y is not None:
    ax.text(99, ood_label_y, "OUT-OF-DISTRIBUTION", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#1a7a1a")

ax.set_title("ReCOGS in-distribution vs per-category generalization: graph (FLOP-matched) vs hypergraph\n"
             f"(graph: {len(gf_dirs)} seeds, hypergraph: {len(hg_dirs)} seeds; error bars = seed std)",
             fontsize=12)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()

out = os.path.join(HERE, "..", "figures", "graph_vs_hypergraph_gen_test_v4.png")
fig.savefig(out, dpi=150)
print("wrote", out)
print("test EM: graph %.1f%% (n=%d)  hypergraph %.1f%% (n=%d)"
      % (np.mean(gf_test), len(gf_test), np.mean(hg_test), len(hg_test)))
