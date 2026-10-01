"""Aggregate results/*.jsonl into tables (LaTeX + JSON) and figures."""
import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
CONDS = ["closed", "single", "agent", "oracle"]
CNAME = {"closed": "Closed-book", "single": "Single-shot RAG", "agent": "Agentic search", "oracle": "Oracle abstracts"}
os.makedirs("paper/figs", exist_ok=True)


def macro_f1(g, p):
    fs = []
    for l in LABELS:
        tp = sum(1 for a, b in zip(g, p) if a == l and b == l)
        fp = sum(1 for a, b in zip(g, p) if a != l and b == l)
        fn = sum(1 for a, b in zip(g, p) if a == l and b != l)
        fs.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def boot(g, p, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    g, p = np.array(g), np.array(p)
    vals = [fn(g[i], p[i]) for i in (rng.integers(0, len(g), len(g)) for _ in range(B))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


acc = lambda g, p: float(np.mean(np.array(g) == np.array(p)))

rows = defaultdict(dict)
for path in glob.glob("results/*.jsonl"):
    model, cond = os.path.basename(path)[:-6].split("__")
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows[model][cond] = {r["id"]: r for r in rs}

# Closed-book, single-shot, and agent are scored on the claims finished in all three.
# The oracle is scored on the claims it completed; the other harnesses are also
# reported on that same subset.
summary = {}
for model, conds in rows.items():
    ids = set.intersection(*[set(conds[c]) for c in ("closed", "single", "agent") if c in conds])
    oids = ids & set(conds.get("oracle", {}))
    summary[model] = {"n": len(ids), "n_oracle": len(oids)}
    if oids:
        summary[model]["oracle_subset"] = {}
        for cond, rs in conds.items():
            rr = [rs[i] for i in sorted(oids)]
            g, p = [r["gold"] for r in rr], [r["label"] for r in rr]
            summary[model]["oracle_subset"][cond] = {"acc": acc(g, p), "f1": macro_f1(g, p)}
    for cond, rs in conds.items():
        rr = [rs[i] for i in sorted(oids if cond == "oracle" else ids)]
        g, p = [r["gold"] for r in rr], [r["label"] for r in rr]
        s = {"acc": acc(g, p), "acc_ci": boot(g, p, acc), "f1": macro_f1(g, p), "f1_ci": boot(g, p, macro_f1),
             "pred_dist": dict(Counter(p)),
             "per_label_acc": {l: acc([x for x in g if x == l], [b for a, b in zip(g, p) if a == l]) if l in g else None for l in LABELS}}
        if cond in ("single", "agent"):
            ev = [r for r in rr if r["gold"] != "NEI"]
            s["gold_recall"] = float(np.mean([r["gold_retrieved"] for r in ev])) if ev else None
            s["avg_searches"] = float(np.mean([r["n_search"] for r in rr]))
            s["avg_retrieved"] = float(np.mean([len(r["retrieved"]) for r in rr]))
            s["zero_hit_rate"] = float(np.mean([len(r["retrieved"]) == 0 for r in rr]))
            cited = [r for r in rr if r["cited"]]
            s["unfaithful_cite_rate"] = float(np.mean([bool(r["cited_unretrieved"]) for r in cited])) if cited else 0.0
            s["n_with_citations"] = len(cited)
            # accuracy conditioned on whether gold evidence was retrieved
            for flag in (True, False):
                sub = [r for r in ev if r["gold_retrieved"] == flag]
                s[f"acc_ev_gold_{'hit' if flag else 'miss'}"] = acc([r["gold"] for r in sub], [r["label"] for r in sub]) if sub else None
                s[f"n_ev_gold_{'hit' if flag else 'miss'}"] = len(sub)
        if cond == "closed":
            cited = [r for r in rr if r["cited"]]
            s["n_with_citations"] = len(cited)
            s["cite_nonexistent_rate"] = float(np.mean([len(r.get("cited_exists", [])) < len(r["cited"]) for r in cited])) if cited else 0.0
            s["cite_gold_rate"] = float(np.mean([r.get("cited_is_gold", False) for r in cited])) if cited else 0.0
            s["n_cited_pmids"] = sum(len(r["cited"]) for r in cited)
            s["n_cited_pmids_exist"] = sum(len(r.get("cited_exists", [])) for r in cited)
        summary[model][cond] = s
    # paired comparison agent vs single and agent vs closed (McNemar-style counts)
    for a, b in (("agent", "single"), ("agent", "closed"), ("single", "closed")):
        if a in conds and b in conds:
            ca = [conds[a][i]["label"] == conds[a][i]["gold"] for i in sorted(ids)]
            cb = [conds[b][i]["label"] == conds[b][i]["gold"] for i in sorted(ids)]
            summary[model][f"{a}_vs_{b}"] = {"a_only": int(sum(x and not y for x, y in zip(ca, cb))),
                                            "b_only": int(sum(y and not x for x, y in zip(ca, cb)))}

json.dump(summary, open("results/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

# ---------------- LaTeX main table
MNAME = {"gemini-3.1-flash-lite": "Gemini-3.1-Flash-Lite", "gemini-3.5-flash-lite": "Gemini-3.5-Flash-Lite"}
lines = []
for model in sorted(summary):
    S = summary[model]
    for i, cond in enumerate(CONDS):
        if cond not in S:
            continue
        s = S[cond]
        rec = f"{100 * s['gold_recall']:.0f}" if s.get("gold_recall") is not None else "--"
        srch = f"{s['avg_searches']:.2f}" if "avg_searches" in s else "--"
        m = f"\\multirow{{4}}{{*}}{{{MNAME.get(model, model)}}}" if i == 0 else ""
        partial = cond == "oracle" and S["n_oracle"] < S["n"]
        cname = CNAME[cond] + (f"$^\\dagger$ {{\\scriptsize($n{{=}}{S['n_oracle']}$)}}" if partial else "")
        lines.append(f"{m} & {cname} & {100 * s['acc']:.1f} {{\\scriptsize[{100 * s['acc_ci'][0]:.0f},{100 * s['acc_ci'][1]:.0f}]}} & "
                     f"{100 * s['f1']:.1f} {{\\scriptsize[{100 * s['f1_ci'][0]:.0f},{100 * s['f1_ci'][1]:.0f}]}} & {rec} & {srch} \\\\")
    lines.append("\\midrule")
head = ("\\begin{tabular}{llcccc}\n\\toprule\nBackbone & Harness & Acc. (\\%) & Macro-F1 (\\%) & Gold recall (\\%) & \\#Searches \\\\\n\\midrule\n")
open("paper/table_main.tex", "w").write(head + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- figures
models = sorted(summary)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
w = 0.8 / max(1, len(models))
for k, model in enumerate(models):
    S = summary[model]
    xs = np.arange(len(CONDS))
    vals = [100 * S[c]["f1"] if c in S else 0 for c in CONDS]
    lo = [100 * (S[c]["f1"] - S[c]["f1_ci"][0]) if c in S else 0 for c in CONDS]
    hi = [100 * (S[c]["f1_ci"][1] - S[c]["f1"]) if c in S else 0 for c in CONDS]
    axes[0].bar(xs + k * w - 0.4 + w / 2, vals, w, yerr=[lo, hi], capsize=2, label=MNAME.get(model, model))
axes[0].set_xticks(np.arange(len(CONDS)))
axes[0].set_xticklabels(["Closed", "Single\nRAG", "Agentic", "Oracle"], fontsize=8)
axes[0].set_ylabel("Macro-F1 (%)")
axes[0].legend(fontsize=7, loc="upper left")
axes[0].set_title("(a) Verdict quality", fontsize=9)

for k, model in enumerate(models):
    S = summary[model]
    vals, labs = [], []
    for c in ("single", "agent"):
        if c in S:
            for key in ("hit", "miss"):
                v = S[c].get(f"acc_ev_gold_{key}")
                vals.append(100 * v if v is not None else 0)
    xs = np.arange(len(vals))
    axes[1].bar(xs + k * w - 0.4 + w / 2, vals, w, label=MNAME.get(model, model))
axes[1].set_xticks(np.arange(4))
axes[1].set_xticklabels(["RAG\ngold hit", "RAG\ngold miss", "Agent\ngold hit", "Agent\ngold miss"], fontsize=8)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("(b) Accuracy on SUP/CON claims\nby whether gold paper was retrieved", fontsize=9)
plt.tight_layout()
plt.savefig("paper/figs/main.pdf")
print("wrote paper/table_main.tex and paper/figs/main.pdf")
