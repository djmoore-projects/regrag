# RegRAG — cited answers over U.S. financial regulations

[![Tests](https://github.com/djmoore-projects/regrag/actions/workflows/tests.yml/badge.svg)](https://github.com/djmoore-projects/regrag/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

A knowledge assistant over **Regulation Z (Truth in Lending), Regulation E (Electronic Fund Transfers), and the Bank
Secrecy Act rules (31 CFR 1010 / 1020)**, built from the official eCFR XML. Every sentence in an answer links to
the exact regulatory paragraph that supports it, e.g. `§ 1026.5(b)(2)(ii)(A)(1)` or `Comment 54(a)(1)-2.i`,
with the supporting text quoted verbatim.

- **Hybrid retrieval**: pgvector (dense) + BM25 with a citation-aware tokenizer, fused with RRF
- **Cross-encoder reranking**, explicit-citation lookup, and a rule-text guarantee
- **Paragraph-level citations** using Claude's native `search_result` citations, so citations cannot point outside the retrieved context
- **Structure-aware chunking** of hard documents: flat XML with inline `(a)(1)(i)(A)` markers, official commentary, tables
- **Evaluation**: 54-question hand-written golden set, retrieval ablations, and Ragas faithfulness / context recall / answer relevancy with a separate judge model
- **Observability**: OpenTelemetry → Arize Phoenix (retriever, reranker, and LLM spans)
- **FastAPI + simple UI, Dockerized** (Postgres/pgvector + API + Phoenix via Compose)

> This repo previously held an Oracle 23ai / OCI prototype (dense-only, notebooks). It was rebuilt as v2;
> the history is in git. The v1 README's evaluation table was not reproducible and has been removed. Every
> number below comes from a committed result file in [`eval/results/`](eval/results/).

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Ingest (python -m regrag.ingest)"]
        X[eCFR XML<br/>Reg Z, Reg E, 31 CFR 1010/1020] --> P[Parser<br/>rebuilds (a)(1)(i) hierarchy,<br/>comment labels, tables→Markdown]
        P --> C[Structure-aware chunker<br/>13,299 paragraphs → 3,502 chunks]
        C --> E[bge-base-en-v1.5<br/>ONNX, CPU]
    end
    E --> PG[(Postgres + pgvector<br/>HNSW)]
    C --> BM[BM25 index<br/>citation-aware tokenizer]

    subgraph Query["/ask"]
        Q[Question] --> D[Dense top-50]
        Q --> K[BM25 top-50]
        D --> F[RRF fusion]
        K --> F
        F --> R[Cross-encoder rerank<br/>MiniLM-L-12, pool 30<br/>+ linked rule text]
        Q -. explicit citation .-> PIN[Pin cited paragraph]
        R --> G{top score < 0?}
        PIN --> G
        G -- yes --> A0[Abstain<br/>no LLM call]
        G -- no --> LLM[Claude Opus 5.5<br/>search_result blocks,<br/>one per paragraph]
        LLM --> OUT[Answer + paragraph-level citations]
    end
    PG --> D
    BM --> K
    Query -. OTel spans .-> PH[Arize Phoenix]
```

## Why this corpus is hard

eCFR is a good "messy real corpus" because nothing in it is chunk-friendly:

| Problem | Example | What we do |
|---|---|---|
| Hierarchy exists only as inline text markers | `<P>(3)(i) Application means…</P>`, `<P>(11) <I>Due date.</I> (i) Except…</P>` | A marker tracker rebuilds the path (`§ 1026.2(a)(3)(i)`), including back-to-back and italic-heading markers and the `(i)` letter-vs-roman ambiguity. 0 malformed labels across 13,299 paragraphs. |
| Two-thirds of the text is Official Interpretation (commentary) with its own citation scheme | `Comment 54(a)(1)-2.ii.A`; Reg Z and Reg E use different heading layouts | Parsed into CFPB-style comment labels (97% of commentary paragraphs labelled), linked back to the rule section they interpret |
| Commentary crowds out the rule it explains | For "When is the Loan Estimate due?" the top results were all `Comment 19(e)-…` | **Rule-text guarantee**: if the top-k has commentary on a section but none of its rule text, the best-scoring rule chunk for that section is swapped in |
| Users cite paragraphs | "What does § 1026.54(b) say?" | The BM25 tokenizer emits `1026.54`, `1026.54(b)`, … as tokens, and explicit citations are pinned before semantic ranking. Dense-only recall on these questions: **0.0** |
| Tables and model forms | Appendix H ARM examples, penalty tables | Rendered to Markdown and kept as atomic blocks |
| Point-in-time law vs. model memory | Late-fee safe harbor is **$8** in the current eCFR text (2024 rule), not the $30/$41 a model may remember | Answers come only from retrieved text; this case is in the golden set |

### Chunking strategy

A chunk never crosses a section boundary and is built from whole paragraphs. Paragraphs are grouped by top-level
paragraph (`§ 1026.54(a)`) or by commentary heading (`Comment 54(a)(1)`). Groups over 300 words are split at
paragraph boundaries; small neighbours in the same section are merged. Every chunk gets a breadcrumb header
(`Regulation Z (Truth in Lending) › § 1026.54 … › § 1026.54(a)(1) – 1026.54(b)(2)`) that is embedded and indexed
with the text, so a fragment like "(ii) Any portion of a balance…" stays findable. Each paragraph stays a separate
**block**, and blocks are what Claude cites, so a citation resolves to `§ 1026.54(a)(1)(ii)`, not "chunk 1832".

### Citations you can trust

Retrieved chunks are sent as Claude `search_result` content blocks with citations enabled, one text block per
paragraph. Citations come back as `(search_result_index, start_block_index, end_block_index)`, and the API
guarantees `cited_text` is the verbatim block text. The service maps each citation to paragraph labels and the
eCFR URL and drops any citation it cannot resolve. The UI shows each sentence with its citation chips and the
quoted source. **Citation coverage** (share of answer text attached to a citation) is reported per answer and in
the evaluation.

---

## Evaluation

**Golden set**: [`eval/golden.jsonl`](eval/golden.jsonl). 54 hand-written questions, each with a reference answer
and gold paragraph labels verified against the eCFR text, across these categories: numeric thresholds and deadlines,
lists, rules, definitions, commentary-only answers, multi-hop, exact-citation lookups, a table-like question,
a point-in-time conflict, and **7 out-of-corpus questions** (flood insurance, FDIC limits, CRA, Reg CC, OFAC,
Reg DD, FCRA) that the system should decline.

Relevance is judged on **paragraph labels, not chunk IDs**, so the same golden set can score any chunking strategy.

<!-- RETRIEVAL:START -->
### Retrieval ablation

47 answerable golden questions · embeddings `BAAI/bge-base-en-v1.5` · reranker `Xenova/ms-marco-MiniLM-L-12-v2` (pool 30) · relevance judged on gold paragraph labels · latency is CPU-only on a laptop.

| Configuration | Recall@1 | Recall@3 | Recall@8 | MRR | nDCG@8 | p50 retrieval |
|---|---|---|---|---|---|---|
| Dense only (pgvector) | 0.160 | 0.362 | 0.628 | 0.335 | 0.403 | 0.1 s |
| BM25 only | 0.351 | 0.532 | 0.748 | 0.512 | 0.568 | 0.0 s |
| Hybrid (RRF) | 0.234 | 0.511 | 0.791 | 0.424 | 0.502 | 0.1 s |
| Dense + rerank | 0.394 | 0.745 | 0.809 | 0.588 | 0.646 | 6.8 s |
| Hybrid + rerank | 0.415 | 0.787 | 0.894 | 0.627 | 0.695 | 7.9 s |
| Dense + rerank + structural boosts | 0.404 | 0.787 | 0.872 | 0.612 | 0.682 | 7.2 s |
| **Hybrid + rerank + structural boosts (default)** | 0.426 | 0.809 | 0.894 | 0.634 | 0.704 | 7.5 s |
| Fast profile: `ms-marco-MiniLM-L-6-v2`, pool 20 | 0.383 | 0.670 | 0.904 | 0.585 | 0.672 | 2.6 s |

Recall@8 by question category (dense only → default pipeline):

| Category | n | Dense | Default |
|---|---|---|---|
| definition | 2 | 0.500 | 1.000 |
| exact-citation | 2 | 0.000 | 1.000 |
| interpretation | 3 | 0.667 | 0.833 |
| list | 7 | 0.571 | 0.857 |
| multi-hop | 4 | 0.875 | 0.875 |
| numeric | 20 | 0.700 | 0.900 |
| numeric-conflict | 1 | 1.000 | 1.000 |
| rule | 7 | 0.571 | 0.857 |
| table-like | 1 | 0.000 | 1.000 |

**Abstention.** Top rerank score, answerable vs out-of-corpus: min answerable 0.40; out-of-corpus -6.72, -6.19, -4.08, -2.87, -2.09, 3.52, 4.90. The balanced-accuracy optimum (3.57) would refuse 9% of answerable questions, so the shipped threshold is 0.0: no answerable question is refused, 5/7 out-of-corpus questions are declined before any LLM call, and the rest rely on the model declining.
<!-- RETRIEVAL:END -->

### End-to-end (Ragas)

[`eval/rag_eval.py`](eval/rag_eval.py) runs the full pipeline and scores it with **Ragas 0.4** (faithfulness,
context recall, answer relevancy). The judge is `claude-sonnet-5-5` and the generator is `claude-opus-5-5`, so no
model grades its own answers. Claude structured outputs back a small Ragas LLM adapter. It also measures citation
coverage and abstention: out-of-corpus questions should be declined, answerable ones should not.

<!-- RAG:START -->
_Pending: the end-to-end run needs an Anthropic API key and is the next step. No numbers are shown until a real run produces them._
<!-- RAG:END -->

### Honest caveats

- **Small golden set.** 47 answerable questions is enough to see large effects (dense → hybrid + rerank:
  Recall@8 0.63 → 0.89) but not to resolve differences of a few points. The structural boosts (linked rule
  text, rule-text guarantee, citation pinning) were designed after reading the misses in the first ablation, on the
  same questions. Their measured effect is small (one question fixed, one regressed, small MRR/Recall@3 gains) and
  in-sample. A held-out set is the next step.
- **Latency** is CPU-only on a laptop: the cross-encoder dominates (~7.5 s p50 for 30 candidates). The fast profile
  (MiniLM-L-6, pool 20) runs in ~2.6 s with lower Recall@3/MRR (see table). A GPU or hosted reranker removes most of this.
- **Not legal advice.** Answers reflect eCFR text as of the ingest date (`ECFR_DATE`, default 2026-10-01).

---

## Observability

With `PHOENIX_COLLECTOR_ENDPOINT` set (Compose does this), every `/ask` produces a trace:
`ask` → `retrieve` (`dense_search`, `bm25_search`, `rerank` with document IDs and scores) → `generate` → the
Anthropic call (auto-instrumented via OpenInference: model, tokens, latency). Answers also record
`regrag.abstained` and `regrag.citation_coverage`. Open Phoenix at http://localhost:6006.

## Run it

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up -d            # postgres+pgvector, api, phoenix
docker compose run --rm api python -m regrag.ingest   # download eCFR, chunk, embed (one-time, CPU)
open http://localhost:8000      # UI; API docs at /docs
```

Local development:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,eval]"
docker run -d --name regrag-db -e POSTGRES_USER=regrag -e POSTGRES_PASSWORD=regrag -e POSTGRES_DB=regrag -p 5432:5432 pgvector/pgvector:pg17
.venv/bin/python -m regrag.ingest
.venv/bin/uvicorn regrag.api:app --reload
```

### API

| Endpoint | Purpose |
|---|---|
| `POST /ask` `{question, mode?, k?}` | Grounded answer: `segments` (text + citation numbers), `citations` (labels, verbatim quote, eCFR URL), `abstained`, `citation_coverage`, retrieval debug, `trace_id` |
| `POST /search` | Retrieval only (no LLM): ranked chunks with dense / BM25 / fused ranks and rerank scores |
| `GET /health` | Readiness + chunk count |

`mode` ∈ `dense`, `bm25`, `hybrid`, `dense_rerank`, `hybrid_rerank` (default). The UI exposes it so the
ablation can be reproduced interactively.

**Cost guards for a public demo**: per-IP rate limit and a global daily cap on LLM calls (`RATE_LIMIT_PER_MINUTE`,
`DAILY_LLM_BUDGET`). Abstentions and `/search` don't count toward either, because they never reach the LLM.

### Reproduce the evaluation

```bash
.venv/bin/python -m eval.retrieval_eval --out eval/results/retrieval_nolink.json --no-link   # baseline modes
.venv/bin/python -m eval.retrieval_eval --modes dense_rerank hybrid_rerank --out eval/results/retrieval_link.json
.venv/bin/python -m eval.rag_eval --limit 8          # stratified smoke run (~$1)
.venv/bin/python -m eval.rag_eval                    # full run, both configs
.venv/bin/python -m eval.report                      # regenerates eval/results/REPORT.md
```

## Project layout

```
regrag/
  ecfr.py         eCFR download + XML parser (marker hierarchy, commentary labels, tables)
  chunking.py     structure-aware chunker (blocks = citable paragraphs)
  bm25.py         BM25 with citation/alias-aware tokenizer
  store.py        Postgres + pgvector (HNSW, cosine)
  retrieval.py    dense + BM25 → RRF → rerank, citation pinning, rule-text guarantee
  generation.py   Claude search_result citations → paragraph-labelled citations; abstention
  tracing.py      OpenTelemetry → Phoenix (OpenInference conventions)
  api.py          FastAPI service + rate limiting; static/index.html UI
eval/
  golden.jsonl    54 hand-written questions with gold paragraph labels
  retrieval_eval.py, rag_eval.py, report.py, results/
tests/            offline unit tests (no DB, models, or API key)
docker/Dockerfile, docker-compose.yml
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | Postgres 17 + pgvector (HNSW) | One database for chunks, metadata, and vectors. Runs anywhere (the v1 Oracle 23ai design required an OCI tenancy) |
| Keyword | `bm25s` + PyStemmer, custom tokenizer | True BM25, plus legal-citation tokens that generic analyzers destroy |
| Embeddings / rerank | `fastembed` ONNX: bge-base-en-v1.5, ms-marco-MiniLM-L-12 | No GPU, no extra API keys, baked into the image |
| Generation | Claude Opus 5.5 with native search-result citations, server-side refusal fallback | Verbatim, block-level citations |
| Evaluation | Ragas 0.4 with a Claude Sonnet 5.5 judge | Faithfulness, context recall, answer relevancy |
| Tracing | OpenTelemetry → Arize Phoenix | Self-hosted in Compose, no account needed |
| Serving | FastAPI + vanilla JS UI, Docker Compose | |

---

Built by [Derek Moore](mailto:derek@aismartr.com) · AI Solutions Engineer
