"""Hybrid retrieval: dense (pgvector) + BM25, fused with RRF, then cross-encoder rerank."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .bm25 import BM25Index
from .chunking import Chunk
from .config import Settings, get_settings
from .evaluation import label_matches
from .store import Store
from .tracing import set_documents, set_io, tracer

# BGE v1.5 retrieval instruction for queries (passages are embedded without it).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODES = ("dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank")
_COMMENT_ANCHOR = re.compile(r"^Comment \d+((?:\([A-Za-z0-9]+\))*)-")
_QUERY_CFR = re.compile(r"(\d{3,4}\.\d+)((?:\([A-Za-z0-9]+\))*)")
_QUERY_COMMENT = re.compile(r"comment\s+(\d+(?:\([A-Za-z0-9]+\))*-\d+(?:\.[A-Za-z0-9]+)*)", re.I)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    query: str
    mode: str
    hits: list[Hit]
    candidates: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_score(self) -> float | None:
        return self.hits[0].rerank_score if self.hits and self.hits[0].rerank_score is not None else None


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Embedder:
    def __init__(self, model: str) -> None:
        self.model = TextEmbedding(model, threads=os.cpu_count())

    def embed_passages(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        # Small batches: ONNX pads each batch to its longest text, so big batches waste CPU.
        return np.array(list(self.model.embed(texts, batch_size=batch_size)))

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self.model.embed([QUERY_PREFIX + text])))


class Retriever:
    def __init__(self, store: Store, settings: Settings | None = None,
                 embedder: Embedder | None = None) -> None:
        import time

        self.settings = settings or get_settings()
        self.store = store
        self.embedder = embedder or Embedder(self.settings.embed_model)
        self.reranker = TextCrossEncoder(self.settings.rerank_model)
        t0 = time.perf_counter()
        self.chunks: dict[str, Chunk] = {c.id: c for c in store.all_chunks()}
        ids = list(self.chunks)
        self.bm25 = BM25Index(ids, [self.chunks[i].search_text for i in ids])
        self.reg_by_section: dict[str, list[Chunk]] = {}
        for c in self.chunks.values():
            if c.kind == "regulation":
                self.reg_by_section.setdefault(c.section_id, []).append(c)
        self.load_ms = (time.perf_counter() - t0) * 1000

    # -- individual retrievers -------------------------------------------------
    def dense(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.store.dense_search(self.embedder.embed_query(query), k)

    def keyword(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, k)

    def rerank(self, query: str, ids: list[str]) -> list[tuple[str, float]]:
        texts = [self.chunks[i].search_text for i in ids]
        scores = list(self.reranker.rerank(query, texts, batch_size=16))
        return sorted(zip(ids, scores, strict=False), key=lambda x: x[1], reverse=True)

    def linked_regulation(self, chunk: Chunk, limit: int = 2) -> list[str]:
        """Regulation chunks that an Official Interpretation chunk interprets.

        "Comment 19(e)-5" in section 1026.19 → chunks covering § 1026.19(e). If no chunk
        matches the full paragraph path, back off one level at a time.
        """
        if chunk.kind != "interpretation" or not chunk.blocks:
            return []
        m = _COMMENT_ANCHOR.match(chunk.blocks[0].label)
        candidates = self.reg_by_section.get(chunk.section_id, [])
        if not m or not candidates:
            return []
        parts = re.findall(r"\([A-Za-z0-9]+\)", m.group(1))
        while parts:
            target = f"§ {chunk.section_id}{''.join(parts)}"
            found = [c.id for c in candidates if any(label_matches(b.label, target) for b in c.blocks)]
            if found:
                return found[:limit]
            parts.pop()
        return []

    def cited_chunks(self, query: str, limit: int = 2) -> list[str]:
        """Chunks containing a paragraph the question cites explicitly ("§ 1026.54(b)",
        "comment 52(b)(2)(i)-1"). A cross-encoder can't read citations, so these are pinned."""
        targets = [f"§ {base}{subs}" for base, subs in _QUERY_CFR.findall(query)]
        targets += [f"Comment {c}" for c in _QUERY_COMMENT.findall(query)]
        out: list[str] = []
        for t in targets:
            for c in self.chunks.values():
                if c.id not in out and any(label_matches(b.label, t) for b in c.blocks):
                    out.append(c.id)
                    break  # first (earliest) chunk covering the cited paragraph
        return out[:limit]

    def ensure_rule_text(self, ranked: list[tuple[str, float]], k: int, max_swaps: int = 2):
        """If the top-k holds commentary on a section but none of that section's rule text,
        swap the section's best-reranked regulation chunk in for the lowest-ranked hit."""
        top = list(ranked[:k])
        rest = ranked[k:]
        swaps = 0
        for d, _ in list(top):
            c = self.chunks[d]
            if swaps >= max_swaps or c.kind != "interpretation":
                continue
            if any(self.chunks[x].kind == "regulation" and self.chunks[x].section_id == c.section_id
                   for x, _ in top):
                continue
            cand = next(((x, sc) for x, sc in rest if self.chunks[x].kind == "regulation"
                         and self.chunks[x].section_id == c.section_id
                         and sc >= self.settings.abstain_threshold), None)
            if cand and len(top) == k:
                rest.remove(cand)
                top[len(top) - 1 - swaps] = cand
                swaps += 1
        return sorted(top, key=lambda x: x[1], reverse=True)

    # -- pipeline --------------------------------------------------------------
    def retrieve(self, query: str, mode: str = "hybrid_rerank", k: int | None = None) -> RetrievalResult:
        import time

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        s = self.settings
        k = k or s.final_k
        n = s.candidates_per_retriever
        timings: dict[str, float] = {}

        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("retrieval.mode", mode)
            set_io(span, query)

            dense_hits: list[tuple[str, float]] = []
            bm25_hits: list[tuple[str, float]] = []
            if mode in ("dense", "hybrid", "dense_rerank", "hybrid_rerank"):
                t = time.perf_counter()
                with tracer.start_as_current_span("dense_search") as sp:
                    sp.set_attribute("openinference.span.kind", "RETRIEVER")
                    dense_hits = self.dense(query, n)
                    set_documents(sp, "retrieval.documents", [(i, sc, "") for i, sc in dense_hits[:10]])
                timings["dense_ms"] = (time.perf_counter() - t) * 1000
            if mode in ("bm25", "hybrid", "hybrid_rerank"):
                t = time.perf_counter()
                with tracer.start_as_current_span("bm25_search") as sp:
                    sp.set_attribute("openinference.span.kind", "RETRIEVER")
                    bm25_hits = self.keyword(query, n)
                    set_documents(sp, "retrieval.documents", [(i, sc, "") for i, sc in bm25_hits[:10]])
                timings["bm25_ms"] = (time.perf_counter() - t) * 1000

            dense_rank = {doc_id: r for r, (doc_id, _) in enumerate(dense_hits, 1)}
            bm25_rank = {doc_id: r for r, (doc_id, _) in enumerate(bm25_hits, 1)}

            if mode in ("hybrid", "hybrid_rerank"):
                fused = rrf([[d for d, _ in dense_hits], [d for d, _ in bm25_hits]], k=s.rrf_k)
            elif mode in ("dense", "dense_rerank"):
                fused = dense_hits
            else:
                fused = bm25_hits
            fused_rank = {doc_id: r for r, (doc_id, _) in enumerate(fused, 1)}

            if mode.endswith("_rerank"):
                pool = [d for d, _ in fused[: s.rerank_pool]]
                if s.structural_boosts:
                    # Commentary dominates the corpus; make sure the rule text it
                    # interprets is in the pool so the reranker can choose it.
                    extra: list[str] = []
                    for d in pool[:10]:
                        for linked in self.linked_regulation(self.chunks[d]):
                            if linked not in pool and linked not in extra:
                                extra.append(linked)
                    pool += extra[: s.link_expansion_max]
                t = time.perf_counter()
                with tracer.start_as_current_span("rerank") as sp:
                    sp.set_attribute("openinference.span.kind", "RERANKER")
                    sp.set_attribute("reranker.model_name", s.rerank_model)
                    ranked = self.rerank(query, pool)
                    set_documents(sp, "reranker.output_documents",
                                  [(i, sc, self.chunks[i].breadcrumb) for i, sc in ranked[:k]])
                timings["rerank_ms"] = (time.perf_counter() - t) * 1000
                top = self.ensure_rule_text(ranked, k) if s.structural_boosts else ranked[:k]
                pinned = self.cited_chunks(query) if s.structural_boosts else []
                if pinned:
                    scores = dict(ranked)
                    best = max((sc for _, sc in ranked), default=0.0)
                    pins = [(d, max(scores.get(d, best), best)) for d in pinned]
                    top = (pins + [x for x in top if x[0] not in pinned])[:k]
                hits = [
                    Hit(self.chunks[d], sc, dense_rank.get(d), bm25_rank.get(d), fused_rank.get(d), sc)
                    for d, sc in top
                ]
            else:
                hits = [
                    Hit(self.chunks[d], sc, dense_rank.get(d), bm25_rank.get(d), fused_rank.get(d))
                    for d, sc in fused[:k]
                ]

            set_documents(span, "retrieval.documents",
                          [(h.chunk.id, h.score, h.chunk.search_text) for h in hits])
            return RetrievalResult(query, mode, hits, candidates=len(fused), timings_ms=timings)
