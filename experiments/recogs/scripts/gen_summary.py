"""Emit a compact per-category gen + overall test summary for a set of run dirs.

Usage: python gen_summary.py <base_dir> <glob>   (e.g. . 'runs/v3/graph_flash_*')
Prints one line of JSON: {"n": seeds, "per_cat": {cat: [acc% per seed]}, "test": [acc% per seed]}.
Kept tiny so it can be run on any pod and the JSON pulled back over actl.
"""
import glob, json, collections, os, sys

base, pat = sys.argv[1], sys.argv[2]
dirs = sorted(os.path.dirname(p)
              for p in glob.glob(os.path.join(base, pat, "per_example_gen.json")))
per = {}
for d in dirs:
    rows = json.load(open(os.path.join(d, "per_example_gen.json")))
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["category"]][0] += r["correct"]
        agg[r["category"]][1] += 1
    for c, (cor, tot) in agg.items():
        per.setdefault(c, []).append(100.0 * cor / tot)
test = []
for d in dirs:
    p = os.path.join(d, "per_example_test.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        if rows:
            test.append(100.0 * sum(r["correct"] for r in rows) / len(rows))
print(json.dumps({"n": len(dirs), "per_cat": per, "test": test}))
