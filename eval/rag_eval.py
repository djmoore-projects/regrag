"""End-to-end RAG evaluation with Ragas, judged by a different model than the generator.

    python -m eval.rag_eval --modes dense hybrid_rerank [--limit N] [--regenerate]

Step 1 runs the full pipeline over the golden set and caches answers to
eval/results/answers_<mode>.jsonl (so re-judging never re-generates).
Step 2 scores them:
  * Ragas: faithfulness, context recall, answer relevancy (answerable items)
  * Citation coverage: share of answer text attached to a citation
  * Abstention: out-of-corpus questions should be declined, answerable ones not
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import anthropic
import numpy as np
from pydantic import BaseModel
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

from regrag.config import get_settings
from regrag.evaluation import load_golden
from regrag.generation import Generator
from regrag.retrieval import Embedder, Retriever
from regrag.store import Store

RESULTS = Path(__file__).parent / "results"
GOLDEN = Path(__file__).parent / "golden.jsonl"
JUDGE_MODEL = "claude-sonnet-5-5"


class ClaudeJudge(InstructorBaseRagasLLM):
    """Ragas LLM adapter using Claude structured outputs (messages.parse)."""

    def __init__(self, model: str = JUDGE_MODEL, concurrency: int = 4) -> None:
        self.model = model
        self.client = anthropic.AsyncAnthropic(max_retries=6)
        self.sem = asyncio.Semaphore(concurrency)

    async def agenerate(self, prompt: str, response_model):
        async with self.sem:
            resp = await self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                output_config={"effort": "medium"},
                messages=[{"role": "user", "content": prompt}],
                output_format=response_model,
            )
        return resp.parsed_output

    def generate(self, prompt: str, response_model):
        return asyncio.get_event_loop().run_until_complete(self.agenerate(prompt, response_model))


class FastEmbedRagas(BaseRagasEmbedding):
    def __init__(self, embedder: Embedder) -> None:
        super().__init__()
        self.embedder = embedder

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return np.asarray(self.embedder.embed_passages([text])[0]).tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)


class DeclineCheck(BaseModel):
    declines: bool  # True if the answer says the sources don't contain the answer


DECLINE_PROMPT = (
    "Does the following answer decline to answer, or state that the provided sources do "
    "not contain the requested information? Partial answers that give substantive "
    "regulatory content count as NOT declining.\n\nANSWER:\n{answer}"
)


def generate_answers(mode: str, limit: int | None, regenerate: bool) -> list[dict]:
    path = RESULTS / f"answers_{mode}.jsonl"
    items = load_golden(GOLDEN)
    if limit:  # stratified: round-robin across categories so a smoke run covers every kind
        by_cat: dict[str, list] = {}
        for i in items:
            by_cat.setdefault(i.category, []).append(i)
        picked = []
        while len(picked) < limit and any(by_cat.values()):
            for cat in sorted(by_cat):
                if by_cat[cat] and len(picked) < limit:
                    picked.append(by_cat[cat].pop(0))
        items = picked
    cached = {}
    if path.exists() and not regenerate:
        cached = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    todo = [i for i in items if i.id not in cached]
    if todo:
        s = get_settings()
        retriever = Retriever(Store(s.database_url, s.embed_dim), s)
        gen = Generator(s)
        for it in todo:
            res = retriever.retrieve(it.question, mode)
            ans = gen.answer(it.question, res)
            cached[it.id] = {
                "id": it.id, "category": it.category, "question": it.question,
                "reference": it.reference, "answerable": it.answerable, "mode": mode,
                "answer": ans.answer, "abstained": ans.abstained,
                "citation_coverage": ans.citation_coverage,
                "citations": [asdict(c) for c in ans.citations],
                "contexts": [h.chunk.search_text for h in res.hits],
                "top_rerank_score": res.top_score, "usage": ans.usage, "timings_ms": ans.timings_ms,
            }
            print(f"[{mode}] {it.id}: {'ABSTAIN' if ans.abstained else ans.answer[:80]!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                for r in cached.values():
                    f.write(json.dumps(r) + "\n")
    return [cached[i.id] for i in items]


async def score(rows: list[dict], judge: ClaudeJudge, emb: FastEmbedRagas) -> list[dict]:
    faith, recall, relev = Faithfulness(llm=judge), ContextRecall(llm=judge), AnswerRelevancy(llm=judge, embeddings=emb)

    async def one(r: dict) -> dict:
        out = {"id": r["id"], "category": r["category"], "answerable": r["answerable"]}
        if r["abstained"]:
            declined = True
        else:
            dc = await judge.agenerate(DECLINE_PROMPT.format(answer=r["answer"]), DeclineCheck)
            declined = dc.declines
        out["declined"] = declined
        out["citation_coverage"] = r["citation_coverage"]
        if r["answerable"]:
            recall_res = await recall.ascore(user_input=r["question"], retrieved_contexts=r["contexts"],
                                             reference=r["reference"])
            out["context_recall"] = float(recall_res.value)
            if not declined:
                f = await faith.ascore(user_input=r["question"], response=r["answer"],
                                       retrieved_contexts=r["contexts"])
                a = await relev.ascore(user_input=r["question"], response=r["answer"])
                out["faithfulness"] = float(f.value)
                out["answer_relevancy"] = float(a.value)
        return out

    return list(await asyncio.gather(*(one(r) for r in rows)))


def summarize(mode: str, scored: list[dict], rows: list[dict]) -> dict:
    ans = [s for s in scored if s["answerable"]]
    un = [s for s in scored if not s["answerable"]]

    def mean(key, pool):
        vals = [s[key] for s in pool if s.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    answered = [s for s in ans if not s["declined"]]
    usage = [r["usage"] for r in rows if r["usage"]]
    llm_ms = [r["timings_ms"].get("llm_ms") for r in rows if r["timings_ms"].get("llm_ms")]
    return {
        "mode": mode,
        "n": len(scored),
        "faithfulness": mean("faithfulness", answered),
        "context_recall": mean("context_recall", ans),
        "answer_relevancy": mean("answer_relevancy", answered),
        "citation_coverage": mean("citation_coverage", answered),
        "answer_rate_answerable": round(len(answered) / len(ans), 3) if ans else None,
        "abstain_rate_unanswerable": round(sum(s["declined"] for s in un) / len(un), 3) if un else None,
        "avg_input_tokens": round(statistics.mean(u["input_tokens"] for u in usage)) if usage else None,
        "avg_output_tokens": round(statistics.mean(u["output_tokens"] for u in usage)) if usage else None,
        "llm_latency_p50_ms": round(statistics.median(llm_ms)) if llm_ms else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["dense", "hybrid_rerank"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    judge = ClaudeJudge()
    emb = FastEmbedRagas(Embedder(s.embed_model))
    summary = {"generator": s.anthropic_model, "judge": JUDGE_MODEL, "results": []}
    for mode in args.modes:
        rows = generate_answers(mode, args.limit, args.regenerate)
        scored = asyncio.run(score(rows, judge, emb))
        (RESULTS / f"scores_{mode}.jsonl").write_text("\n".join(json.dumps(x) for x in scored) + "\n")
        summ = summarize(mode, scored, rows)
        summary["results"].append(summ)
        print(json.dumps(summ, indent=2))
    out = RESULTS / ("rag_eval.json" if not args.limit else "rag_eval_smoke.json")
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
