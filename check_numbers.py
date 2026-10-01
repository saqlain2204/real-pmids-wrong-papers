"""Recompute every number stated in the paper text from raw results and compare with the claimed value."""
import json
from collections import Counter

M = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
R = {(m, c): [json.loads(l) for l in open(f"results/{m}__{c}.jsonl", encoding="utf-8")] for m in M for c in ("closed", "single", "agent", "oracle")}
S = json.load(open("results/summary.json"))
MC = json.load(open("results/mcnemar.json"))
CJ = json.load(open("results/citation_judge.json"))
JS = json.load(open("results/judge_sanity.json"))

checks = []


def chk(desc, claimed, actual):
    ok = str(claimed) == str(actual)
    checks.append(ok)
    print(f"{'OK ' if ok else 'BAD'} {desc}: paper={claimed} data={actual}")


pct = lambda x: round(100 * x)
pct1 = lambda x: f"{100 * x:.1f}"
gold = Counter(r["gold"] for r in R[(M[0], "closed")])
chk("claims per condition", "100,100,100,100", ",".join(str(len(R[(M[0], c)])) for c in ("closed", "single", "agent", "oracle")))
chk("evidence-bearing claims", 64, gold["SUPPORT"] + gold["CONTRADICT"])
chk("gold NEI share %", 36, gold["NEI"])
for i, m in enumerate(M):
    s = S[m]
    chk(f"{m} closed acc", ["55.0", "59.0"][i], pct1(s["closed"]["acc"]))
    chk(f"{m} single acc", ["53.0", "49.0"][i], pct1(s["single"]["acc"]))
    chk(f"{m} agent acc", "62.0", pct1(s["agent"]["acc"]))
    chk(f"{m} oracle acc", ["86.0", "79.0"][i], pct1(s["oracle"]["acc"]))
    chk(f"{m} agent F1", ["60.3", "61.1"][i], pct1(s["agent"]["f1"]))
    chk(f"{m} oracle F1", ["85.8", "78.6"][i], pct1(s["oracle"]["f1"]))
    chk(f"{m} oracle-agent gap (pts)", [24, 17][i], round(100 * (s["oracle"]["acc"] - s["agent"]["acc"])))
    chk(f"{m} single gold recall hits", "4/64", f"{s['single']['n_ev_gold_hit']}/64")
    chk(f"{m} single gold recall %", 6, pct(s["single"]["gold_recall"]))
    chk(f"{m} agent gold recall %", [19, 20][i], pct(s["agent"]["gold_recall"]))
    chk(f"{m} single NEI prediction %", [59, 70][i], s["single"]["pred_dist"]["NEI"])
    chk(f"{m} single acc when gold retrieved", "1.0", s["single"]["acc_ev_gold_hit"])
    chk(f"{m} avg searches", ["1.42", "1.70"][i], f"{s['agent']['avg_searches']:.2f}")
    ns = Counter(r["n_search"] for r in R[(m, "agent")])
    chk(f"{m} single-search stop %", [71, 55][i], ns[1])
    chk(f"{m} full-budget %", [3, 6][i], ns[4])
    chk(f"{m} agent acc on gold-miss SUP/CON %", 75, pct(s["agent"]["acc_ev_gold_miss"]))
    chk(f"{m} agent NEI recall %", [33, 42][i], pct(s["agent"]["per_label_acc"]["NEI"]))
    chk(f"{m} single NEI recall %", [72, 81][i], pct(s["single"]["per_label_acc"]["NEI"]))
    chk(f"{m} oracle NEI recall %", [78, 86][i], pct(s["oracle"]["per_label_acc"]["NEI"]))
    chk(f"{m} closed NEI prediction %", [4, 9][i], s["closed"]["pred_dist"]["NEI"])
    chk(f"{m} closed NEI recall %", [6, 17][i], pct(s["closed"]["per_label_acc"]["NEI"]))
    chk(f"{m} agent unfaithful-cite %", ["0.0", "2.6"][i], pct1(s["agent"]["unfaithful_cite_rate"]))
    chk(f"{m} gold-hit n (fig caption)", [12, 13][i], s["agent"]["n_ev_gold_hit"])
    c = CJ[m]
    chk(f"{m} closed claims with citations", [97, 80][i], c["claims_with_citations"])
    chk(f"{m} closed existing/cited", ["186/190", "141/145"][i], f"{c['existing']}/{c['cited_pmids']}")
    chk(f"{m} closed relevant", [1, 2][i], c["relevant"])
    a = CJ[m + "__agent"]
    chk(f"{m} agent relevant/cited", ["141/215", "128/184"][i], f"{a['relevant']}/{a['cited_pmids']}")
    chk(f"{m} agent relevant %", [66, 70][i], pct(a["relevant"] / a["cited_pmids"]))
    chk(f"{m} agent claims w/ any relevant %", [77, 84][i], pct(a["claims_with_any_relevant"] / a["claims_with_citations"]))
chk("pooled agent relevant %", 67, pct((CJ[M[0] + "__agent"]["relevant"] + CJ[M[1] + "__agent"]["relevant"]) / (CJ[M[0] + "__agent"]["cited_pmids"] + CJ[M[1] + "__agent"]["cited_pmids"])))
chk("pooled closed exist", "327/335", f"{CJ[M[0]]['existing'] + CJ[M[1]]['existing']}/{CJ[M[0]]['cited_pmids'] + CJ[M[1]]['cited_pmids']}")
chk("judge sanity", "21/25,0/25", f"{JS['gold_judged_relevant']}/{JS['n_gold']},{JS['mismatched_judged_relevant']}/{JS['n_mismatched']}")
for key, claimed in [("gemini-3.1-flash-lite closed_vs_single", "28-26 p=0.89"), ("gemini-3.5-flash-lite closed_vs_single", "34-24 p=0.24"),
                     ("gemini-3.1-flash-lite single_vs_agent", "15-24 p=0.2"), ("gemini-3.5-flash-lite single_vs_agent", "16-29 p=0.07"),
                     ("gemini-3.1-flash-lite closed_vs_agent", "6-13 p=0.17"), ("gemini-3.5-flash-lite closed_vs_agent", "8-11 p=0.65"),
                     ("gemini-3.1-flash-lite agent_vs_oracle", "5-29 p=0.0"), ("gemini-3.5-flash-lite agent_vs_oracle", "9-26 p=0.006")]:
    v = MC[key]
    chk(key, claimed, f"{v['a_only']}-{v['b_only']} p={round(v['p'], 3 if v['p'] < 0.01 else 2)}")
print(f"\n{sum(checks)}/{len(checks)} checks passed")
