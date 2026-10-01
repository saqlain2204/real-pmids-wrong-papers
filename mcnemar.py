"""Exact two-sided McNemar tests for all harness pairs."""
import json
from itertools import combinations
from math import comb

CONDS = ["closed", "single", "agent", "oracle"]
out = {}
for m in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
    R = {c: {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{m}__{c}.jsonl", encoding="utf-8")} for c in CONDS}
    ids = sorted(set.intersection(*[set(v) for v in R.values()]))
    for a, b in combinations(CONDS, 2):
        ca = [R[a][i]["label"] == R[a][i]["gold"] for i in ids]
        cb = [R[b][i]["label"] == R[b][i]["gold"] for i in ids]
        x = sum(p and not q for p, q in zip(ca, cb))
        y = sum(q and not p for p, q in zip(ca, cb))
        pval = min(1.0, 2 * sum(comb(x + y, k) for k in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0
        out[f"{m} {a}_vs_{b}"] = {"a_only": x, "b_only": y, "p": round(pval, 4)}
        print(f"{m:24s} {a:6s} vs {b:6s}: {x:2d} vs {y:2d}  p={pval:.4f}")
json.dump(out, open("results/mcnemar.json", "w"), indent=2)
