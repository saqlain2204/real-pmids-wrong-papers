"""Closed-book vs. retrieval vs. agentic PubMed search for biomedical claim verification (SciFact dev)."""
import argparse
import difflib
import hashlib
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DATA = "data/data"
OUT = "results"
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
MAX_AGENT_STEPS = 4
TOPK = 5

os.makedirs(OUT, exist_ok=True)
os.makedirs("cache", exist_ok=True)


# ---------------------------------------------------------------- caching
class JsonCache:
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.d = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        k, v = json.loads(line)
                        self.d[k] = v
                    except Exception:
                        pass

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps([k, v]) + "\n")


LLM_CACHE = JsonCache("cache/llm.jsonl")
PM_CACHE = JsonCache("cache/pubmed.jsonl")


# ---------------------------------------------------------------- PubMed tools
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ncbi_lock = threading.Lock()
_ncbi_last = [0.0]


def _ncbi_get(endpoint, params):
    for attempt in range(6):
        with _ncbi_lock:
            wait = 0.36 - (time.time() - _ncbi_last[0])
            if wait > 0:
                time.sleep(wait)
            _ncbi_last[0] = time.time()
        try:
            r = requests.get(f"{EUTILS}/{endpoint}", params=params, timeout=30)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NCBI failed: {endpoint} {params}")


def pubmed_fetch(pmids):
    """Return {pmid: {title, abstract}} for a list of PMIDs."""
    out, missing = {}, []
    for p in pmids:
        c = PM_CACHE.get("doc:" + p)
        if c is not None:
            out[p] = c
        else:
            missing.append(p)
    if missing:
        r = _ncbi_get("efetch.fcgi", {"db": "pubmed", "id": ",".join(missing), "retmode": "xml"})
        root = ET.fromstring(r.content)
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            abst = " ".join("".join(a.itertext()) for a in art.findall(".//Abstract/AbstractText"))
            doc = {"title": title.strip(), "abstract": abst.strip()}
            PM_CACHE.put("doc:" + pmid, doc)
            out[pmid] = doc
        for p in missing:
            if p not in out:
                PM_CACHE.put("doc:" + p, {"title": "", "abstract": ""})
                out[p] = {"title": "", "abstract": ""}
    return out


def pubmed_search(query, k=TOPK):
    key = f"search:{k}:{query}"
    ids = PM_CACHE.get(key)
    if ids is None:
        r = _ncbi_get("esearch.fcgi", {"db": "pubmed", "term": query, "retmax": k, "sort": "relevance", "retmode": "json"})
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        PM_CACHE.put(key, ids)
    docs = pubmed_fetch(ids) if ids else {}
    return [(p, docs[p]) for p in ids if p in docs]


STOP = set("a an the of in on and or to for with by is are was were be been as at from that this these those than via its it into not no".split())


def naive_query(claim):
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", claim) if w.lower() not in STOP]
    return " ".join(words)


def single_shot_retrieve(claim):
    q = naive_query(claim)
    hits = pubmed_search(q)
    if not hits:  # PubMed ANDs all terms; relax to OR over content words
        hits = pubmed_search(" OR ".join(q.split()))
    return q, hits


def fmt_docs(hits, maxlen=1500):
    return "\n\n".join(f"[PMID {p}] {d['title']}\n{d['abstract'][:maxlen]}" for p, d in hits)


# ---------------------------------------------------------------- LLM backends
def call_llm(model, system, user):
    key = hashlib.sha256(json.dumps([model, system, user]).encode()).hexdigest()
    c = LLM_CACHE.get(key)
    if c is not None:
        return c
    for attempt in range(8):
        try:
            if model.startswith("gemini"):
                txt = _gemini(model, system, user)
            else:
                txt = _groq(model, system, user)
            LLM_CACHE.put(key, txt)
            return txt
        except RateLimit as e:
            time.sleep(min(60, 5 * (attempt + 1)))
        except Exception as e:
            if attempt >= 5:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("LLM failed")


class RateLimit(Exception):
    pass


def _gemini(model, system, user):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    r = requests.post(url, json=body, timeout=120)
    if r.status_code in (429, 503):
        raise RateLimit(r.text[:200])
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _groq(model, system, user):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, timeout=120)
    if r.status_code in (429, 503):
        raise RateLimit(r.text[:200])
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json(txt):
    txt = re.sub(r"^```(json)?|```$", "", txt.strip()).strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def norm_label(x):
    x = str(x or "").upper()
    if x.startswith("SUP"):
        return "SUPPORT"
    if x.startswith("CON") or x.startswith("REF"):
        return "CONTRADICT"
    return "NEI"


# ---------------------------------------------------------------- prompts
TASK = (
    "You are a biomedical fact-checking assistant. Decide whether the scientific literature SUPPORTS "
    "the claim, CONTRADICTS it, or provides NOT ENOUGH INFO (NEI). Use NEI when the evidence does not directly "
    "address the claim."
)
ANSWER_FMT = (
    'Respond with JSON: {"label": "SUPPORT"|"CONTRADICT"|"NEI", "cited_pmids": [list of PubMed IDs (strings) '
    'of the papers your verdict relies on], "rationale": "<= 2 sentences"}'
)


def run_closed(model, claim):
    sys_ = TASK + " You have no access to tools; rely on your own knowledge. If you know specific papers, cite their PubMed IDs. " + ANSWER_FMT
    out = parse_json(call_llm(model, sys_, f"Claim: {claim}"))
    return {"label": norm_label(out.get("label")), "cited": [str(x) for x in out.get("cited_pmids", []) or []],
            "retrieved": [], "n_search": 0, "queries": []}


def run_single(model, claim):
    q, hits = single_shot_retrieve(claim)
    sys_ = TASK + " Base your verdict only on the retrieved abstracts below; cite only PMIDs that appear in them. " + ANSWER_FMT
    out = parse_json(call_llm(model, sys_, f"Claim: {claim}\n\nRetrieved abstracts:\n{fmt_docs(hits) or '(no results)'}"))
    return {"label": norm_label(out.get("label")), "cited": [str(x) for x in out.get("cited_pmids", []) or []],
            "retrieved": [p for p, _ in hits], "n_search": 1, "queries": [q]}


AGENT_SYS = TASK + f""" You can search PubMed. PubMed search is keyword-based (terms are ANDed), so short focused queries of 2-6 key terms work best; you may use OR and field tags like [tiab]. You have at most {MAX_AGENT_STEPS} searches.
At each step respond with JSON, either
  {{"action": "search", "query": "<PubMed query>", "reason": "<short>"}}
or, when you have enough evidence (or searches are exhausted),
  {{"action": "answer", "label": "SUPPORT"|"CONTRADICT"|"NEI", "cited_pmids": [PMIDs from the search results only], "rationale": "<= 2 sentences"}}"""


def run_agent(model, claim):
    history, retrieved, queries, seen = [], [], [], set()
    for step in range(MAX_AGENT_STEPS + 1):
        force = step == MAX_AGENT_STEPS
        user = f"Claim: {claim}\n\n" + ("\n\n".join(history) if history else "(no searches yet)")
        if force:
            user += "\n\nSearch budget exhausted. You must now answer."
        out = parse_json(call_llm(model, AGENT_SYS, user))
        if out.get("action") == "search" and not force and out.get("query"):
            q = str(out["query"])[:300]
            queries.append(q)
            hits = pubmed_search(q)
            new = [(p, d) for p, d in hits if p not in seen]
            for p, _ in hits:
                if p not in seen:
                    seen.add(p)
                    retrieved.append(p)
            obs = fmt_docs(new) if new else ("(no new results)" if hits else "(no results; try fewer/broader terms)")
            history.append(f"Search {step + 1}: {q}\nResults:\n{obs}")
            continue
        return {"label": norm_label(out.get("label")), "cited": [str(x) for x in out.get("cited_pmids", []) or []],
                "retrieved": retrieved, "n_search": len(queries), "queries": queries}
    return {"label": "NEI", "cited": [], "retrieved": retrieved, "n_search": len(queries), "queries": queries}


def run_oracle(model, claim, gold_docs):
    sys_ = TASK + " Base your verdict only on the abstracts below. " + ANSWER_FMT
    docs = "\n\n".join(f"[DOC {d['doc_id']}] {d['title']}\n{' '.join(d['abstract'])}" for d in gold_docs)
    out = parse_json(call_llm(model, sys_.replace("PubMed IDs", "DOC ids"), f"Claim: {claim}\n\nAbstracts:\n{docs}"))
    return {"label": norm_label(out.get("label")), "cited": [], "retrieved": [], "n_search": 0, "queries": []}


# ---------------------------------------------------------------- data
def load():
    corpus = {}
    with open(f"{DATA}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["doc_id"]] = d
    claims = [json.loads(l) for l in open(f"{DATA}/claims_dev.jsonl", encoding="utf-8")]
    for c in claims:
        labs = {e["label"] for ev in c["evidence"].values() for e in ev}
        c["gold"] = "NEI" if not labs else ("SUPPORT" if "SUPPORT" in labs else "CONTRADICT")
        c["gold_docs"] = [corpus[int(d)] for d in c["cited_doc_ids"] if int(d) in corpus]
    return claims


def normt(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def title_match(a, b):
    a, b = normt(a), normt(b)
    return bool(a) and bool(b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() > 0.9)


def gold_found(pmids, gold_docs):
    if not pmids or not gold_docs:
        return False
    docs = pubmed_fetch(pmids)
    return any(title_match(docs[p]["title"], g["title"]) for p in pmids if p in docs for g in gold_docs)


# ---------------------------------------------------------------- main
def process(model, cond, c):
    if cond == "closed":
        r = run_closed(model, c["claim"])
    elif cond == "single":
        r = run_single(model, c["claim"])
    elif cond == "agent":
        r = run_agent(model, c["claim"])
    else:
        r = run_oracle(model, c["claim"], c["gold_docs"])
    r.update(id=c["id"], claim=c["claim"], gold=c["gold"], model=model, cond=cond)
    if cond in ("single", "agent"):
        r["gold_retrieved"] = gold_found(r["retrieved"], c["gold_docs"])
        r["cited_unretrieved"] = [p for p in r["cited"] if p not in r["retrieved"]]
    if cond == "closed" and r["cited"]:
        valid = [p for p in r["cited"] if p.isdigit()]
        docs = pubmed_fetch(valid) if valid else {}
        r["cited_exists"] = [p for p in valid if docs.get(p, {}).get("title")]
        r["cited_is_gold"] = any(title_match(docs[p]["title"], g["title"]) for p in r["cited_exists"] for g in c["gold_docs"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemini-3.1-flash-lite,gemini-3.5-flash-lite")
    ap.add_argument("--conds", default="closed,single,agent,oracle")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    claims = load()
    random.Random(0).shuffle(claims)
    claims = claims[: a.n]
    for model in a.models.split(","):
        for cond in a.conds.split(","):
            path = f"{OUT}/{model.replace('/', '_')}__{cond}.jsonl"
            done = set()
            if os.path.exists(path):
                done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
            todo = [c for c in claims if c["id"] not in done]
            print(f"{model} {cond}: {len(done)} done, {len(todo)} todo", flush=True)
            lock = threading.Lock()
            with ThreadPoolExecutor(a.workers) as ex, open(path, "a", encoding="utf-8") as f:
                futs = {ex.submit(process, model, cond, c): c for c in todo}
                for i, fu in enumerate(as_completed(futs)):
                    try:
                        r = fu.result()
                    except Exception as e:
                        print("ERR", futs[fu]["id"], repr(e)[:200], flush=True)
                        continue
                    with lock:
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                    if i % 25 == 0:
                        print(f"  {model} {cond} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
