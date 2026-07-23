#!/usr/bin/env python3
"""Per-category ReCOGS generalization chart, RUN_5 (v5): the full hypergraph
width ladder (3L d128/d256/d384/d512/d768) + the 6L/d256 iso-FLOP depth control,
WITH an in-distribution test-set reference row on the same axis.

Same layout as plots/plot_gen_test_v4.py (test row on top, green lexical band,
structural section, error bars = seed std), but six arms per category: the five
width rungs use a sequential blue ramp (light -> dark = small -> large, so the
ladder ordering is visible at a glance) and the depth control is orange (same
hue as 6L256 in plot_v5_scaling.py). d256 pools exp/v3_prefix_lm/runs + exp/v5_hg_scaling/runs seeds.

Saves figures/v5_gen_test_per_category_ladder.png.
"""
import glob, json, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

# arm -> (list of run-dir globs, color). Width rungs: sequential blues
# (lightness monotone = ladder order); 6L256 depth control: orange.
ARMS = [
    ("3L/d128",        ["exp/v5_hg_scaling/runs/hg_d128_*"],                                     "#7fb9da"),
    ("3L/d256",        ["exp/v3_prefix_lm/runs/hg_gather_*",
                        "exp/v5_hg_scaling/runs/hg_gather_hypergraph_cuda_*"],                   "#529dcc"),
    ("3L/d384",        ["exp/v5_hg_scaling/runs/hg_d384_*"],                                     "#2e7ebc"),
    ("3L/d512",        ["exp/v5_hg_scaling/runs/hg_d512_*"],                                     "#125da6"),
    ("3L/d768",        ["exp/v5_hg_scaling/runs/hg_d768_*"],                                     "#083c7d"),
    ("6L/d256 (ctrl)", ["exp/v5_hg_scaling/runs/hg_6L256_*"],                                    "#ff7f0e"),
]


def gen_dirs(globs):
    """Run dirs (per seed) that have a per_example_gen.json, sorted, de-duped."""
    out = set()
    for g in globs:
        for p in glob.glob(os.path.join(ROOT, g, "per_example_gen.json")):
            out.add(os.path.dirname(p))
    return sorted(out)


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
    """[overall test EM% per seed] (skip dirs lacking per_example_test.json)."""
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

arm_dirs = {name: gen_dirs(gs) for name, gs, _ in ARMS}
arm_cat = {name: gen_per_category(arm_dirs[name]) for name, _, _ in ARMS}
arm_test = {name: test_overall(arm_dirs[name]) for name, _, _ in ARMS}

all_cats = set().union(*arm_cat.values())
# Sort by the biggest width rung (d768) so the ladder's endpoint orders the rows.
_by_d768 = lambda c: np.mean(arm_cat["3L/d768"].get(c, [0]))
lexical = sorted(all_cats - STRUCTURAL, key=_by_d768, reverse=True)
structural = sorted(all_cats & STRUCTURAL, key=_by_d768, reverse=True)


def stat(name, c):
    v = arm_cat[name].get(c, [0])
    return np.mean(v), np.std(v)


sections = [
    ("IN-DISTRIBUTION (test)", "#000", ["__test__"]),
    ("lexical generalization", "#555", lexical),
    ("structural generalization", "#555", structural),
]

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
                         [(np.mean(arm_test[n]), np.std(arm_test[n])) for n, _, _ in ARMS]))
        else:
            rows.append((label(k), [stat(n, k) for n, _, _ in ARMS]))
        y.append(cur)
        cur += 1.0
    cur += GAP
y = np.array(y)

n_arms = len(ARMS)
h = 0.8 / n_arms  # bar height; the group spans 0.8 of a row
fig, ax = plt.subplots(figsize=(13, 20))

n_test = len(sections[0][2])
n_lex = len(lexical)
if n_lex:
    ax.axhspan(y[n_test] - 0.5, y[n_test + n_lex - 1] + 0.5, color="#bfe3bf", alpha=0.8, zorder=0)

for i, (name, _, color) in enumerate(ARMS):
    means = [r[1][i][0] for r in rows]
    errs = [r[1][i][1] for r in rows]
    off = (i - (n_arms - 1) / 2) * h
    ax.barh(y + off, means, h * 0.92, xerr=errs,
            label=f"{name} (n={len(arm_dirs[name])})", color=color,
            error_kw=dict(ecolor="#444", lw=0.6, capsize=1.5), zorder=3)

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

ax.set_title("ReCOGS in-distribution vs per-category generalization: hypergraph width ladder (RUN_5)\n"
             "(3L width rungs light→dark blue; orange = 6L/d256 iso-FLOP depth control; "
             "error bars = seed std)", fontsize=12)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()

out = os.path.join(HERE, "..", "figures", "v5_gen_test_per_category_ladder.png")
fig.savefig(out, dpi=150)
print("wrote", out)
for name, _, _ in ARMS:
    t = arm_test[name]
    print("  %-16s n=%2d  test %.1f%%" % (name, len(arm_dirs[name]), np.mean(t) if t else float("nan")))
