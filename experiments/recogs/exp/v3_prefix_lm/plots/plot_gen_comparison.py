#!/usr/bin/env python3
"""Plot graph_flash vs hypergraph generalization accuracy per ReCOGS category.

Reads per_example_gen.json from completed runs, averages across seeds (showing
per-seed spread as error bars), and saves a grouped bar chart PNG to docs/.
"""
import glob, json, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

# v3: auto-discover every seed under exp/v3_prefix_lm/runs (10 seeds per arm, pulled from both
# pods). Sorted so the per-seed order is stable across arms.
RUNS_DIR = os.environ.get("RUNS_DIR", "exp/v3_prefix_lm/runs")
ARMS = {
    "hypergraph": sorted(glob.glob(os.path.join(ROOT, RUNS_DIR, "hg_gather_*", "per_example_gen.json"))),
    "graph_flash": sorted(glob.glob(os.path.join(ROOT, RUNS_DIR, "graph_flash_*", "per_example_gen.json"))),
}


def per_category(paths):
    """Return {category: [acc% per seed]} and overall mean accuracy %."""
    per_seed = []
    tot = cor = 0
    for p in paths:
        rows = json.load(open(p))  # absolute paths from glob
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            agg[r["category"]][0] += r["correct"]
            agg[r["category"]][1] += 1
            cor += r["correct"]
            tot += 1
        per_seed.append({c: v[0] / v[1] * 100 for c, v in agg.items()})
    cats = set().union(*[s.keys() for s in per_seed])
    by_cat = {c: [s[c] for s in per_seed if c in s] for c in cats}
    return by_cat, cor / tot * 100


# Human-readable descriptions of each ReCOGS generalization split.
# Convention: "seen as X in training -> tested as Y" (lexical), or a novel
# structural configuration (structural). See Kim & Linzen (COGS) / ReCOGS.
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


def label(c):
    return LABELS.get(c, c)


# The 3 COGS/ReCOGS structural-generalization cases; everything else is lexical.
STRUCTURAL = {"obj_pp_to_subj_pp", "pp_recursion", "cp_recursion"}

hg, hg_overall = per_category(ARMS["hypergraph"])
gf, gf_overall = per_category(ARMS["graph_flash"])
all_cats = set(hg) | set(gf)
_by_gf = lambda c: np.mean(gf.get(c, [0]))
lexical = sorted(all_cats - STRUCTURAL, key=_by_gf, reverse=True)
structural = sorted(all_cats & STRUCTURAL, key=_by_gf, reverse=True)
cats = lexical + structural  # lexical block on top, structural block below


def stats(d, c):
    v = d.get(c, [0])
    return np.mean(v), np.std(v)  # mean, std across seeds for err bar


gf_mean = [stats(gf, c)[0] for c in cats]
gf_err = [stats(gf, c)[1] for c in cats]
hg_mean = [stats(hg, c)[0] for c in cats]
hg_err = [stats(hg, c)[1] for c in cats]

# y positions with a gap between the lexical and structural blocks.
GAP = 1.2
y = []
cur = 0.0
for i, c in enumerate(cats):
    if i == len(lexical):  # first structural row -> insert a gap
        cur += GAP
    y.append(cur)
    cur += 1.0
y = np.array(y)

h = 0.4
fig, ax = plt.subplots(figsize=(13, 9))

# shade the structural block
if structural:
    s_top = y[len(lexical)] - 0.5
    s_bot = y[-1] + 0.5
    ax.axhspan(s_top, s_bot, color="#bfe3bf", alpha=0.45, zorder=0)

ax.barh(y + h / 2, gf_mean, h, xerr=gf_err, label="graph_flash",
        color="#1f77b4", error_kw=dict(ecolor="#444", lw=0.8, capsize=2), zorder=3)
ax.barh(y - h / 2, hg_mean, h, xerr=hg_err, label="hypergraph",
        color="#d62728", error_kw=dict(ecolor="#444", lw=0.8, capsize=2), zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([label(c) for c in cats], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("exact-match accuracy (%)")
ax.set_xlim(0, 100)
ax.set_ylim(y[-1] + 1.0, y[0] - 1.4)  # room for section headers (axis inverted)

# section headers
ax.text(99, y[0] - 1.0, "LEXICAL",
        ha="right", va="center", fontsize=10, fontweight="bold", color="#333")
ax.text(99, y[len(lexical)] - 1.0, "STRUCTURAL",
        ha="right", va="center", fontsize=10, fontweight="bold", color="#1a7a1a")

ax.set_title("ReCOGS generalization by category: graph_flash vs hypergraph\n"
             f"(graph_flash: {len(ARMS['graph_flash'])} seeds, hypergraph: {len(ARMS['hypergraph'])} seeds; "
             "error bars = seed std)", fontsize=12)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()

out = os.path.join(HERE, "..", "figures", "graph_vs_hypergraph_gen_v3.png")
fig.savefig(out, dpi=150)
print("wrote", out)
print(f"overall: graph_flash {gf_overall:.1f}%  hypergraph {hg_overall:.1f}%")
