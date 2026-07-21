#!/usr/bin/env python3
"""Combined RUN_3 (param-matched) + RUN_4 (FLOP-matched) per-category ReCOGS
generalization chart, same layout as plot_gen_with_test_v3.py.

THREE series, hypergraph shown ONCE (it is the identical hg_gather 3L run in both
comparisons, so it is not duplicated):
  - hypergraph (hg_gather 3L, 3.15M)              — red
  - graph param-matched (graph_flash 4L, ~3.2M)   — orange   [RUN_3, equal SIZE]
  - graph FLOP-matched (6L/d1088, ~74M)           — blue     [RUN_4, equal COMPUTE]

Reads three compact summary JSONs (produced by scripts/gen_summary.py):
  SUMDIR/sum_hg.json  sum_graph_pm.json  sum_graph_flop.json
each = {"n": seeds, "per_cat": {cat: [acc% per seed]}, "test": [acc% per seed]}.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMDIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "graph_vs_hypergraph_gen_test_v3v4.png"

hg = json.load(open(os.path.join(SUMDIR, "sum_hg.json")))
pm = json.load(open(os.path.join(SUMDIR, "sum_graph_pm.json")))
fl = json.load(open(os.path.join(SUMDIR, "sum_graph_flop.json")))

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


def stat(summary, c):
    v = summary["per_cat"].get(c, [0])
    return np.mean(v), np.std(v)


all_cats = set(hg["per_cat"]) | set(pm["per_cat"]) | set(fl["per_cat"])
# sort lexical rows by the FLOP-matched graph (headline arm), as in the v4 chart
_key = lambda c: np.mean(fl["per_cat"].get(c, [0]))
lexical = sorted(all_cats - STRUCTURAL, key=_key, reverse=True)
structural = sorted(all_cats & STRUCTURAL, key=_key, reverse=True)

sections = [
    ("IN-DISTRIBUTION (test)", "#000", ["__test__"]),
    ("lexical generalization", "#555", lexical),
    ("structural generalization", "#555", structural),
]

# rows: (label, (hg_m,hg_e),(pm_m,pm_e),(fl_m,fl_e))
rows, headers = [], []
y, cur, GAP = [], 0.0, 1.6
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
                         (np.mean(hg["test"]), np.std(hg["test"])),
                         (np.mean(pm["test"]), np.std(pm["test"])),
                         (np.mean(fl["test"]), np.std(fl["test"]))))
        else:
            rows.append((label(k), stat(hg, k), stat(pm, k), stat(fl, k)))
        y.append(cur)
        cur += 1.0
    cur += GAP
y = np.array(y)

hg_m = [r[1][0] for r in rows]; hg_e = [r[1][1] for r in rows]
pm_m = [r[2][0] for r in rows]; pm_e = [r[2][1] for r in rows]
fl_m = [r[3][0] for r in rows]; fl_e = [r[3][1] for r in rows]

h = 0.26
fig, ax = plt.subplots(figsize=(13.5, 11))

n_test = len(sections[0][2]); n_lex = len(lexical)
if n_lex:
    ax.axhspan(y[n_test] - 0.5, y[n_test + n_lex - 1] + 0.5, color="#bfe3bf", alpha=0.8, zorder=0)

ekw = dict(ecolor="#444", lw=0.7, capsize=2)
# three bars per row; top->bottom within a group: FLOP-matched, param-matched, hypergraph
ax.barh(y + h, fl_m, h, xerr=fl_e, label="graph FLOP-matched (6L/d1088, ~74M) — equal compute",
        color="#1f77b4", error_kw=ekw, zorder=3)
ax.barh(y,      pm_m, h, xerr=pm_e, label="graph param-matched (4L, ~3.2M) — equal size",
        color="#ff7f0e", error_kw=ekw, zorder=3)
ax.barh(y - h,  hg_m, h, xerr=hg_e, label="hypergraph (hg_gather 3L, 3.15M)",
        color="#d62728", error_kw=ekw, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("exact-match accuracy (%)")
ax.set_xlim(0, 100)
ax.set_ylim(y[-1] + 1.0, y[0] - 1.6)

for hy, text, color in headers:
    ax.text(99, hy, text, ha="right", va="center", fontsize=10, fontweight="bold", color=color)
if ood_label_y is not None:
    ax.text(99, ood_label_y, "OUT-OF-DISTRIBUTION", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#1a7a1a")

ax.set_title("ReCOGS per-category generalization — param-matched (RUN_3) vs FLOP-matched (RUN_4) graph vs hypergraph\n"
             f"(hg/pm/flop: {hg['n']}/{pm['n']}/{fl['n']} seeds; hypergraph shared across both comparisons; error bars = seed std)",
             fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
for nm, s in (("hypergraph", hg), ("graph param-matched", pm), ("graph FLOP-matched", fl)):
    allcat = np.concatenate([v for v in s["per_cat"].values()])
    print(f"  {nm:22s}: test {np.mean(s['test']):.1f}%  | gen macro-avg over cats {allcat.mean():.1f}%")
