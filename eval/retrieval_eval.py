"""Retrieval ablation over the golden set (no LLM calls, deterministic).

    python -m eval.retrieval_eval [--rerank-model NAME] [--out eval/results/retrieval.json]

Reports recall@k / hit@k / MRR / nDCG@8 per retrieval mode, per-category
breakdown for the default mode, retrieval latency, and calibrates the
abstention threshold on the top rerank score (answerable vs out-of-corpus).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from regrag.config import get_settings
from regrag.evaluation import load_golden, retrieval_metrics
from regrag.retrieval import MODES, Retriever
from regrag.store import Store

GOLDEN = Path(__file__).parent / "golden.jsonl"


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def calibrate(answerable: list[float], unanswerable: list[float]) -> dict:
    """Pick the threshold that maximises balanced accuracy of abstention."""
    best = None
    for t in sorted(set(answerable + unanswerable)) + [max(answerable + unanswerable) + 1]:
        tpr = sum(s < t for s in unanswerable) / len(unanswerable)  # correctly abstain
        tnr = sum(s >= t for s in answerable) / len(answerable)  # correctly answer
        bal = (tpr + tnr) / 2
        # prefer the lowest threshold among ties: answering is the costlier miss here
        if best is None or bal > best["balanced_accuracy"]:
            best = {"threshold": t, "balanced_accuracy": bal,
                    "abstain_rate_unanswerable": tpr, "answer_rate_answerable": tnr}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank-model")
    parser.add_argument("--out", default="eval/results/retrieval.json")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--rerank-pool", type=int)
    parser.add_argument("--no-link", action="store_true", help="disable structural boosts (baseline)")
    args = parser.parse_args()

    s = get_settings()
    if args.rerank_model:
        s = s.model_copy(update={"rerank_model": args.rerank_model})
    if args.rerank_pool:
        s = s.model_copy(update={"rerank_pool": args.rerank_pool})
    if args.no_link:
        s = s.model_copy(update={"structural_boosts": False})
    retriever = Retriever(Store(s.database_url, s.embed_dim), s)
    items = load_golden(GOLDEN)
    answerable = [i for i in items if i.answerable]

    report: dict = {"rerank_model": s.rerank_model, "embed_model": s.embed_model,
                    "rerank_pool": s.rerank_pool, "structural_boosts": s.structural_boosts,
                    "n_answerable": len(answerable), "n_unanswerable": len(items) - len(answerable),
                    "modes": {}, "per_item": {}}
    for mode in args.modes:
        rows, lat = [], []
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for it in answerable:
            t = time.perf_counter()
            res = retriever.retrieve(it.question, mode, args.k)
            lat.append((time.perf_counter() - t) * 1000)
            m = retrieval_metrics([h.chunk for h in res.hits], it.gold)
            rows.append(m)
            by_cat[it.category].append(m)
            report["per_item"].setdefault(it.id, {})[mode] = {
                "recall@8": m["recall@8"], "mrr": m["mrr"],
                "top": [h.chunk.labels[0] for h in res.hits[:3]],
            }
        agg = {k: round(statistics.mean(r[k] for r in rows), 3) for k in rows[0]}
        agg["latency_p50_ms"] = round(pct(lat, 0.5), 1)
        agg["latency_p95_ms"] = round(pct(lat, 0.95), 1)
        agg["by_category"] = {c: {"n": len(v), "recall@8": round(statistics.mean(x["recall@8"] for x in v), 3),
                                  "mrr": round(statistics.mean(x["mrr"] for x in v), 3)}
                              for c, v in sorted(by_cat.items())}
        report["modes"][mode] = agg
        print(f"{mode:15s} R@1 {agg['recall@1']:.3f} R@3 {agg['recall@3']:.3f} R@8 {agg['recall@8']:.3f} "
              f"MRR {agg['mrr']:.3f} nDCG@8 {agg['ndcg@8']:.3f} p50 {agg['latency_p50_ms']}ms")

    if "hybrid_rerank" not in args.modes:
        Path(args.out).write_text(json.dumps(report, indent=2))
        return
    # Abstention calibration on top rerank score (default pipeline)
    ans_scores = [retriever.retrieve(i.question, "hybrid_rerank", args.k).top_score for i in answerable]
    un_scores = [retriever.retrieve(i.question, "hybrid_rerank", args.k).top_score
                 for i in items if not i.answerable]
    report["abstention"] = {
        "answerable_top_scores": [round(x, 3) for x in ans_scores],
        "unanswerable_top_scores": [round(x, 3) for x in un_scores],
        "calibration": calibrate(ans_scores, un_scores),
    }
    print("abstention:", report["abstention"]["calibration"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
