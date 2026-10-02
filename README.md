# Real PMIDs, Wrong Papers

Code, prompts, raw model outputs, and cached PubMed responses for the paper *Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents* (AgenticLS workshop, NeurIPS 2026).

The study compares four harnesses for biomedical claim verification on a fixed sample of 100 SciFact development claims: closed-book answering, single-shot PubMed retrieval, a ReAct-style agent that writes its own queries, and an oracle given the gold abstracts. Two backbones are used, `gemini-3.1-flash-lite` and `gemini-3.5-flash-lite`.

## Reproduce the reported numbers

The caches and result files are sufficient. No API keys are required.

```bash
pip install -r requirements.txt
python analyze.py
python mcnemar.py
python check_numbers.py
```

`analyze.py` writes `results/summary.json`, `paper/table_main.tex`, and `paper/figs/main.pdf`. `check_numbers.py` recomputes every figure quoted in the paper and compares it with the released results.

## What is included

| Path | Role |
| --- | --- |
| `run_experiment.py` | Harnesses, prompts, PubMed client, and LLM clients |
| `judge_citations.py`, `judge_sanity.py` | Cross-family citation-relevance judge and its controls |
| `analyze.py`, `mcnemar.py`, `check_numbers.py` | Tables, paired tests, and number checks |
| `examples.py` | Prints the qualitative examples discussed in the paper |
| `data/data/claims_dev.jsonl`, `data/data/corpus.jsonl` | SciFact development claims and cited abstracts used here |
| `cache/llm.jsonl`, `cache/pubmed.jsonl`, `cache/judge.jsonl` | Cached model outputs, PubMed responses, and judge decisions |
| `results/*__{closed,single,agent,oracle}.jsonl` | Per-claim outputs for both backbones |

SciFact is the public dataset of Wadden et al. (2020). Only the development claims and the corpus entries needed to recover gold abstracts are included. Cached PubMed records were retrieved from NCBI E-utilities at the time of the experiments.

API keys are read from the environment and are not stored in this repository.

## Optional rerun

Cached calls are reused, so a rerun only contacts an API on a cache miss.

```bash
# PowerShell
$env:GEMINI_API_KEY = "<your key>"
$env:GROQ_API_KEY = "<your key>"   # citation judge only (Qwen via Groq)
python run_experiment.py
python judge_citations.py
python judge_sanity.py
```

`run_experiment.py` defaults to the paper protocol: both backbones, all four harnesses, and the first 100 claims after a shuffle with seed 0. PubMed is queried through NCBI E-utilities (`esearch` with `sort=relevance` and `retmax=5`, then `efetch`). Model calls use temperature 0 and JSON output.

## Citation

```bibtex
@inproceedings{saqlain2026realpmids,
  title     = {Real {PMIDs}, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents},
  author    = {Saqlain, Mohammed},
  booktitle = {Agentic AI for Biological Discovery (AgenticLS) workshop, NeurIPS},
  year      = {2026}
}
```
