"""FastAPI service: /ask (grounded answer), /search (retrieval only), /health, UI at /."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .generation import Generator
from .retrieval import MODES, Retriever
from .store import Store
from .tracing import setup_tracing, tracer

log = logging.getLogger("regrag.api")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: str = "hybrid_rerank"
    k: int = Field(default=8, ge=1, le=15)


class SearchRequest(AskRequest):
    pass


class Limiter:
    """Per-IP sliding-window limit plus a global daily cap on LLM calls (public demo)."""

    def __init__(self, per_minute: int, daily: int) -> None:
        self.per_minute, self.daily = per_minute, daily
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.day_count = date.today(), 0
        self.lock = threading.Lock()

    def check(self, ip: str) -> None:
        now = time.time()
        with self.lock:
            q = self.hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, "Rate limit: try again in a minute.")
            if date.today() != self.day:
                self.day, self.day_count = date.today(), 0
            if self.day_count >= self.daily:
                raise HTTPException(429, "Daily demo budget reached; /search still works.")
            q.append(now)
            self.day_count += 1


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_tracing()
    store = Store(s.database_url, s.embed_dim)
    state["retriever"] = Retriever(store, s)
    state["generator"] = Generator(s)
    state["limiter"] = Limiter(s.rate_limit_per_minute, s.daily_llm_budget)
    log.info("Loaded %d chunks", len(state["retriever"].chunks))
    yield


app = FastAPI(title="RegRAG", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def _hit_dict(h) -> dict:
    return {
        "chunk_id": h.chunk.id, "doc": h.chunk.doc, "kind": h.chunk.kind,
        "breadcrumb": h.chunk.breadcrumb, "url": h.chunk.url, "labels": h.chunk.labels,
        "text": h.chunk.body, "score": h.score, "dense_rank": h.dense_rank,
        "bm25_rank": h.bm25_rank, "fused_rank": h.fused_rank, "rerank_score": h.rerank_score,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    r = state.get("retriever")
    return {"status": "ok" if r else "starting", "chunks": len(r.chunks) if r else 0,
            "model": get_settings().anthropic_model}


@app.post("/search")
def search(req: SearchRequest):
    if req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    res = state["retriever"].retrieve(req.question, req.mode, req.k)
    return {"mode": res.mode, "timings_ms": res.timings_ms, "hits": [_hit_dict(h) for h in res.hits]}


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    if req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    with tracer.start_as_current_span("ask") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", req.question)
        res = state["retriever"].retrieve(req.question, req.mode, req.k)
        top = res.top_score
        if not (top is not None and top < get_settings().abstain_threshold):
            state["limiter"].check(_client_ip(request))
        answer = state["generator"].answer(req.question, res)
        span.set_attribute("output.value", answer.answer)
        span.set_attribute("regrag.abstained", answer.abstained)
        span.set_attribute("regrag.citation_coverage", answer.citation_coverage)
        trace_id = format(span.get_span_context().trace_id, "032x")
    return {
        "question": answer.question,
        "answer": answer.answer,
        "segments": [asdict(s) for s in answer.segments],
        "citations": [asdict(c) for c in answer.citations],
        "abstained": answer.abstained,
        "citation_coverage": round(answer.citation_coverage, 3),
        "model": answer.model,
        "usage": answer.usage,
        "timings_ms": answer.timings_ms,
        "retrieval": {"mode": res.mode, "top_rerank_score": top,
                      "hits": [_hit_dict(h) for h in res.hits]},
        "trace_id": trace_id,
    }
