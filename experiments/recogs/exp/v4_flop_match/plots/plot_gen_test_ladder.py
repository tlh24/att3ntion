#!/usr/bin/env python3
"""ReCOGS per-category generalization ladder, same layout as plot_gen_with_test_v3.py.

FOUR series (hypergraph shown once — identical run across comparisons):
  hypergraph   hg_gather    3L/d256    3.15M   red
  graph        graph_flash  4L/d256    ~3.2M   orange
  graph        graph        6L/d256    ~4.5M   green
  graph        graph        6L/d1088   ~74M    blue

Reads compact summary JSONs (scripts/gen_summary.py) from SUMDIR:
  sum_hg.json sum_graph_pm.json sum_graph_6L256.json sum_graph_flop.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMDIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "graph_vs_hypergraph_gen_test_ladder.png")

# Plot order = top -> bottom within each category group.
SERIES = [
    ("flop", "sum_graph_flop.json",  "graph 6L, d=1088 (~74M)", "#1f77b4"),
    ("d256", "sum_graph_6L256.json", "graph 6L, d=256 (~4.5M)", "#2ca02c"),
    ("pm",   "sum_graph_pm.json",    "graph 4L, d=256 (~3.2M)", "#ff7f0e"),
    ("hg",   "sum_hg.json",          "hypergraph 3L, d=256 (3.15M)", "#d62728"),
]
S = {key: json.load(open(os.path.join(SUMDIR, f))) for key, f, _, _ in SERIES}

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


def stat(key, c):
    v = S[key]["per_cat"].get(c, [0])
    return np.mean(v), np.std(v)


def seed_sigma(summary):
    """Seed-to-seed std (%) of the per-seed overall gen accuracy (macro-avg over
    categories), summarising run-to-run noise for the legend in place of error bars."""
    pc = summary["per_cat"]
    if not pc:
        return 0.0
    n = min(len(v) for v in pc.values())
    per_seed = [float(np.mean([pc[c][i] for c in pc])) for i in range(n)]
    return float(np.std(per_seed))


all_cats = set().union(*[s["per_cat"] for s in S.values()])
_key = lambda c: np.mean(S["flop"]["per_cat"].get(c, [0]))
lexical = sorted(all_cats - STRUCTURAL, key=_key, reverse=True)
structural = sorted(all_cats & STRUCTURAL, key=_key, reverse=True)

sections = [
    ("IN-DISTRIBUTION (test)", "#000", ["__test__"]),
    ("lexical generalization", "#555", lexical),
    ("structural generalization", "#555", structural),
]

rows, headers = [], []   # row = (label, {key:(mean,err)})
y, cur, GAP = [], 0.0, 1.8
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
                         {key: (np.mean(S[key]["test"]), np.std(S[key]["test"])) for key, *_ in SERIES}))
        else:
            rows.append((label(k), {key: stat(key, k) for key, *_ in SERIES}))
        y.append(cur)
        cur += 1.0
    cur += GAP
y = np.array(y)

n = len(SERIES)
h = 0.2
offsets = [(n - 1) / 2.0 - i for i in range(n)]  # series[0] highest offset -> top after invert

fig, ax = plt.subplots(figsize=(14, 12))
n_test = len(sections[0][2]); n_lex = len(lexical)
if n_lex:
    ax.axhspan(y[n_test] - 0.5, y[n_test + n_lex - 1] + 0.5, color="#bfe3bf", alpha=0.8, zorder=0)

for (key, _f, lab, color), off in zip(SERIES, offsets):
    means = [r[1][key][0] for r in rows]
    ax.barh(y + off * h, means, h, label=f"{lab}  ·  seed σ ≈ {seed_sigma(S[key]):.1f}%",
            color=color, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("exact-match accuracy (%)")
ax.set_xlim(0, 100)
ax.set_ylim(y[-1] + 1.0, y[0] - 1.8)

for hy, text, color in headers:
    ax.text(99, hy, text, ha="right", va="center", fontsize=10, fontweight="bold", color=color)
if ood_label_y is not None:
    ax.text(99, ood_label_y, "OUT-OF-DISTRIBUTION", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#1a7a1a")

ax.set_title("ReCOGS per-category OOD generalization: graph vs hypergraph (10 replicates each)", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
for key, _f, lab, _ in SERIES:
    s = S[key]
    a = np.concatenate([v for v in s["per_cat"].values()])
    print(f"  {lab.split(' (')[0]:22s}: test {np.mean(s['test']):.1f}%  gen macro-avg {a.mean():.1f}%")
