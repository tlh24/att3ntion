"""Shared constants + loaders for ReCOGS comparison plots.

Import this from any plot script instead of re-pasting the category labels /
colors / per-example loaders. See docs/PLAYBOOK.md for conventions and which
existing script to use as a template for each plot type.

Data file formats (written by train.py with --eval-at-end):
  eval_curve.tsv         : header + rows  step \\t train_loss \\t dev_loss \\t test_loss
  final_metrics.json     : {split: {"exact_match": frac, "per_category": {cat: {"exact_match": frac}}}}
                           for split in dev/test/gen
  per_example_<split>.json: [{"idx": int, "category": str, "correct": 0/1}, ...]
                           idx is a STABLE global index -> arms align for paired stats.
Run dir naming            : {arm}_{attn_impl}_s{seed}   (e.g. hg_gather_hypergraph_cuda_s3)
"""
import collections
import glob
import json
import os

import numpy as np

# Consistent arm colors across every figure.
COLORS = {
    "hypergraph": "#d62728",   # red
    "graph": "#1f77b4",        # blue (primary graph arm)
    "graph_alt1": "#ff7f0e",   # orange (e.g. param-matched)
    "graph_alt2": "#2ca02c",   # green  (e.g. +depth)
}

# The 3 OOD splits that are *structural* generalization (vs lexical).
STRUCTURAL = {"obj_pp_to_subj_pp", "pp_recursion", "cp_recursion"}

# ReCOGS gen category -> human-readable label. Convention: "seen as X -> tested as Y".
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


def label(cat):
    return LABELS.get(cat, cat)


def run_dirs(base, run_glob):
    """Run dirs under `base` matching `run_glob` that have per_example_gen.json."""
    return sorted(os.path.dirname(p)
                  for p in glob.glob(os.path.join(base, run_glob, "per_example_gen.json")))


def per_category(dirs, split="gen"):
    """{category: [exact-match % per seed]} for a split, over the given run dirs."""
    per_seed = []
    for d in dirs:
        p = os.path.join(d, f"per_example_{split}.json")
        if not os.path.exists(p):
            continue
        agg = collections.defaultdict(lambda: [0, 0])
        for r in json.load(open(p)):
            agg[r["category"]][0] += r["correct"]
            agg[r["category"]][1] += 1
        per_seed.append({c: v[0] / v[1] * 100 for c, v in agg.items()})
    cats = set().union(*[s.keys() for s in per_seed]) if per_seed else set()
    return {c: [s[c] for s in per_seed if c in s] for c in cats}


def overall(dirs, split="test"):
    """[overall exact-match % per seed] for a split (e.g. in-distribution test)."""
    out = []
    for d in dirs:
        p = os.path.join(d, f"per_example_{split}.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        if rows:
            out.append(100 * sum(r["correct"] for r in rows) / len(rows))
    return out


def eval_curves(dirs):
    """[(step, dev_loss, test_loss), ...] per seed; non-finite/≤0 rows dropped.

    Filtering matters: one malformed row in a single seed's tsv otherwise poisons
    the whole arm's min/max and yields an empty plot band.
    """
    out = []
    for d in dirs:
        p = os.path.join(d, "eval_curve.tsv")
        if not os.path.isfile(p):
            continue
        try:
            a = np.genfromtxt(p, delimiter="\t", names=True)
            if a.size == 0:
                continue
            step = np.atleast_1d(a["step"])
            dev = np.atleast_1d(a["dev_loss"])
            test = np.atleast_1d(a["test_loss"])
            ok = np.isfinite(step) & np.isfinite(dev) & np.isfinite(test) & (dev > 0) & (test > 0)
            if ok.sum() >= 2:
                out.append((step[ok], dev[ok], test[ok]))
        except Exception:
            continue
    return out


def mean_band(curves, which="dev"):
    """Mean/std over seeds on a shared grid spanning each arm's FULL step range.

    Uses clamping np.interp (NOT the intersection of step ranges) so an arm whose
    seeds early-stop at different steps still renders a full curve.
    """
    idx = 1 if which == "dev" else 2
    if not curves:
        return None
    lo = min(float(c[0].min()) for c in curves)
    hi = max(float(c[0].max()) for c in curves)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 300)
    M = np.vstack([np.interp(grid, c[0], c[idx]) for c in curves])
    return grid, M.mean(0), M.std(0)
