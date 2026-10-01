"""Sanity check for the relevance judge: gold abstracts (should be relevant) vs. mismatched gold abstracts (should not)."""
import json
import random

from judge_citations import judge
from run_experiment import load

claims = [c for c in load() if c["gold"] != "NEI" and c["gold_docs"]]
random.Random(1).shuffle(claims)
claims = claims[:25]
pos = [judge(c["claim"], {"title": c["gold_docs"][0]["title"], "abstract": " ".join(c["gold_docs"][0]["abstract"])}) for c in claims]
neg = [judge(c["claim"], {"title": claims[(i + 1) % len(claims)]["gold_docs"][0]["title"],
                          "abstract": " ".join(claims[(i + 1) % len(claims)]["gold_docs"][0]["abstract"])}) for i, c in enumerate(claims)]
res = {"gold_judged_relevant": sum(pos), "n_gold": len(pos), "mismatched_judged_relevant": sum(neg), "n_mismatched": len(neg)}
print(res)
json.dump(res, open("results/judge_sanity.json", "w"), indent=2)
