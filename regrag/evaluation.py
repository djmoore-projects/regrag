"""Golden-set loading and label-based relevance for retrieval metrics.

Relevance is judged on paragraph *labels*, not chunk IDs, so the same golden
set scores any chunking strategy. A gold label is a prefix on paragraph
boundaries: gold "§ 1026.13(c)" matches "§ 1026.13(c)(2)" but not "§ 1026.13(cc)";
gold "Comment 6(b)-2" matches "Comment 6(b)-2.i" but not "Comment 6(b)-20".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk


@dataclass
class GoldItem:
    id: str
    category: str
    question: str
    reference: str | None
    gold: list[str]

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def load_golden(path: str | Path) -> list[GoldItem]:
    with open(path) as f:
        return [GoldItem(**json.loads(line)) for line in f if line.strip()]


def label_matches(label: str, gold: str) -> bool:
    label = label.removesuffix(" (table)")
    if not label.startswith(gold):
        return False
    rest = label[len(gold):]
    return rest == "" or rest[0] in "(."


def chunk_gold_hits(chunk: Chunk, gold: list[str]) -> set[str]:
    """Which gold labels does this chunk cover?"""
    return {g for g in gold for b in chunk.blocks if label_matches(b.label, g)}


def retrieval_metrics(ranked: list[Chunk], gold: list[str], ks=(1, 3, 5, 8)) -> dict[str, float]:
    """Recall@k (fraction of gold labels covered in top-k), hit@k, MRR, nDCG@k."""
    rel = [bool(chunk_gold_hits(c, gold)) for c in ranked]
    out: dict[str, float] = {}
    for k in ks:
        covered = set().union(*(chunk_gold_hits(c, gold) for c in ranked[:k])) if ranked[:k] else set()
        out[f"recall@{k}"] = len(covered) / len(gold)
        out[f"hit@{k}"] = float(any(rel[:k]))
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    k = max(ks)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, max(1, sum(rel)))))
    out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return out
