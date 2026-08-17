#!/usr/bin/env python3
"""RUN_7: per-category ReCOGS generalization, graph vs hypergraph at IDENTICAL
shapes (d768: 3L/h12/ffn2304, 10 seeds each side), with an in-distribution
test-set reference row. Same layout as plots/plot_gen_test_v4.py; the v4 figure
compared FLOP-matched graph vs hg — this one compares shape-identical twins at
the top of the RUN_5/RUN_7 width ladder, where the OOD gap is largest.

Saves figures/v7_graph_vs_hg_per_category_parammatch.png.
"""
import glob, json, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

ARM_GLOB = {
    "hypergraph d768 (24.79M)": "exp/v5_hg_scaling/runs/hg_d768_*",
    "graph d870 (24.73M, param-matched)": "exp/v7_graph_ladder/runs/graph_d870_*",
}


def gen_dirs(arm):
    return sorted(
        os.path.dirname(p)
        for p in glob.glob(os.path.join(ROOT, ARM_GLOB[arm], "per_example_gen.json"))
    )


def gen_per_category(dirs):
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
    out = []
    for d in dirs:
        p = os.path.join(d, "per_example_test.json")
        if os.path.exists(p):
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

hg_dirs, gr_dirs = gen_dirs("hypergraph d768 (24.79M)"), gen_dirs("graph d870 (24.73M, param-matched)")
hg, gr = gen_per_category(hg_dirs), gen_per_category(gr_dirs)
hg_test, gr_test = test_overall(hg_dirs), test_overall(gr_dirs)

all_cats = set(hg) | set(gr)
_by_hg = lambda c: np.mean(hg.get(c, [0]))
lexical = sorted(all_cats - STRUCTURAL, key=_by_hg, reverse=True)
structural = sorted(all_cats & STRUCTURAL, key=_by_hg, reverse=True)


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
                         np.mean(gr_test), np.std(gr_test), np.mean(hg_test), np.std(hg_test)))
        else:
            rows.append((label(k), *stat(gr, k), *stat(hg, k)))
        y.append(cur)
        cur += 1.0
    cur += GAP
y = np.array(y)

gr_mean = [r[1] for r in rows]; gr_err = [r[2] for r in rows]
hg_mean = [r[3] for r in rows]; hg_err = [r[4] for r in rows]

h = 0.4
fig, ax = plt.subplots(figsize=(13, 10))

n_test = len(sections[0][2])
n_lex = len(lexical)
if n_lex:
    ax.axhspan(y[n_test] - 0.5, y[n_test + n_lex - 1] + 0.5, color="#bfe3bf", alpha=0.8, zorder=0)

ax.barh(y + h / 2, gr_mean, h, xerr=gr_err, label="graph d870 (param-matched)",
        color="#1f77b4", error_kw=dict(ecolor="#444", lw=0.8, capsize=2), zorder=3)
ax.barh(y - h / 2, hg_mean, h, xerr=hg_err, label="hypergraph d768 (24.79M)",
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

ax.set_title("ReCOGS per-category generalization at PARAM PARITY (~24.7M): graph d870 vs hypergraph d768\n"
             f"(hg: 3L/h12/ffn2304 · graph: 3L/h15/ffn2610; graph: {len(gr_dirs)} seeds, hypergraph: {len(hg_dirs)} seeds; "
             "error bars = seed std)", fontsize=12)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()

out = os.path.join(HERE, "..", "figures", "v7_graph_vs_hg_per_category_parammatch.png")
fig.savefig(out, dpi=150)
print("wrote", out)
print("test EM: graph %.1f%% (n=%d)  hypergraph %.1f%% (n=%d)"
      % (np.mean(gr_test), len(gr_test), np.mean(hg_test), len(hg_test)))
