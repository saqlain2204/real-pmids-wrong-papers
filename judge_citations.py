"""Cross-family relevance judge for PMIDs cited by closed-book models (Qwen via Groq)."""
import glob
import json
import os
import time

import requests

from run_experiment import JsonCache, parse_json, pubmed_fetch

JUDGE = "qwen/qwen3.8-27b"
CACHE = JsonCache("cache/judge.jsonl")
SYS = ("You judge whether a cited paper is relevant evidence for a biomedical claim. Answer 'yes' only if the paper "
       "studies the specific entities and relationship in the claim (so it could plausibly support or refute it); "
       "answer 'no' if it is off-topic or only loosely related. Respond with JSON {\"relevant\": \"yes\"|\"no\"}.")


def judge(claim, doc):
    key = json.dumps([claim, doc["title"]])
    c = CACHE.get(key)
    if c is not None:
        return c
    user = f"Claim: {claim}\n\nCited paper title: {doc['title']}\nAbstract: {doc['abstract'][:700]}"
    for attempt in range(10):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                              json={"model": JUDGE, "temperature": 0, "response_format": {"type": "json_object"},
                                    "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]},
                              timeout=60)
        except requests.RequestException:
            time.sleep(4 * (attempt + 1))
            continue
        if r.status_code == 200:
            v = str(parse_json(r.json()["choices"][0]["message"]["content"]).get("relevant", "no")).lower().startswith("y")
            CACHE.put(key, v)
            return v
        time.sleep(4 * (attempt + 1))
    raise RuntimeError("judge failed")


def main():
    out = {}
    for path in glob.glob("results/*__closed.jsonl"):
        model = os.path.basename(path).split("__")[0]
        rs = [json.loads(l) for l in open(path, encoding="utf-8")]
        n_pmid = n_exist = n_rel = n_claim_any_rel = n_claim_cited = 0
        for r in rs:
            ex = r.get("cited_exists", [])
            if not r["cited"]:
                continue
            n_claim_cited += 1
            n_pmid += len(r["cited"])
            n_exist += len(ex)
            docs = pubmed_fetch(ex) if ex else {}
            rel = [judge(r["claim"], docs[p]) for p in ex if p in docs]
            n_rel += sum(rel)
            n_claim_any_rel += any(rel)
        out[model] = {"claims_with_citations": n_claim_cited, "cited_pmids": n_pmid, "existing": n_exist,
                      "relevant": n_rel, "claims_with_any_relevant": n_claim_any_rel}
        print(model, out[model], flush=True)
    for path in glob.glob("results/*__agent.jsonl") + glob.glob("results/*__single.jsonl"):
        name = os.path.basename(path)[:-6]
        rs = [json.loads(l) for l in open(path, encoding="utf-8")]
        n_pmid = n_rel = n_claim_any_rel = n_claim_cited = 0
        for r in rs:
            cited = [p for p in r["cited"] if p in r["retrieved"]]
            if not cited:
                continue
            n_claim_cited += 1
            docs = pubmed_fetch(cited)
            rel = [judge(r["claim"], docs[p]) for p in cited if p in docs]
            n_pmid += len(rel)
            n_rel += sum(rel)
            n_claim_any_rel += any(rel)
        out[name] = {"claims_with_citations": n_claim_cited, "cited_pmids": n_pmid, "relevant": n_rel,
                     "claims_with_any_relevant": n_claim_any_rel}
        print(name, out[name], flush=True)
    json.dump(out, open("results/citation_judge.json", "w"), indent=2)


if __name__ == "__main__":
    main()
