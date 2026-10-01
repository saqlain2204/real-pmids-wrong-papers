import json

from run_experiment import pubmed_fetch

M = "gemini-3.1-flash-lite"
agent = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{M}__agent.jsonl", encoding="utf-8")}
single = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{M}__single.jsonl", encoding="utf-8")}
closed = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{M}__closed.jsonl", encoding="utf-8")}

print("=== agent found gold, single did not")
for i, a in agent.items():
    if a.get("gold_retrieved") and not single[i].get("gold_retrieved"):
        print(a["claim"], "| gold", a["gold"], "| agent", a["label"], "| single", single[i]["label"])
        print("   single query:", single[i]["queries"])
        print("   agent queries:", a["queries"])
print("\n=== agent missed gold (sample)")
for i, a in list(agent.items())[:40]:
    if a["gold"] != "NEI" and not a.get("gold_retrieved"):
        print(a["claim"], "| gold", a["gold"], "| agent", a["label"], "| q:", a["queries"])
print("\n=== closed-book citations (sample)")
for i, c in list(closed.items())[:6]:
    ex = c.get("cited_exists", [])
    docs = pubmed_fetch(ex) if ex else {}
    print(c["claim"], "| gold", c["gold"], "| pred", c["label"])
    for p in ex:
        print("   PMID", p, "->", docs[p]["title"][:120])
