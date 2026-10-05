"""Grounded answer generation with Claude native search-result citations.

Each retrieved chunk is sent as a ``search_result`` content block whose
``content`` is one text block per regulatory paragraph. Claude's citations
point at (search_result_index, start_block_index, end_block_index), so every
citation maps back to exact paragraph labels such as ``§ 1026.54(a)(1)(ii)``,
and ``cited_text`` is guaranteed by the API to be verbatim source text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

from .config import Settings, get_settings
from .retrieval import Hit, RetrievalResult
from .tracing import set_io, tracer

SYSTEM_PROMPT = """\
You answer questions about U.S. consumer-finance and Bank Secrecy Act regulations \
(Regulation Z, Regulation E, 31 CFR 1010 and 1020) for compliance professionals.

Use only the search results in the user's message. Every factual statement must be \
supported by a search result. If the results do not contain the answer, say so plainly \
and name what is missing instead of answering from general knowledge.

Treat regulation text as the source of requirements, and the Official Interpretations \
(comments) as clarification of it. State numbers, deadlines, and thresholds exactly as \
written, and name the section or comment you rely on (for example § 1026.54(a)(1) or \
comment 54(a)(1)-2). Be concise: lead with the direct answer, then the conditions or \
exceptions that matter."""

NO_ANSWER = (
    "I couldn't find support for this in the indexed regulations (Regulation Z, "
    "Regulation E, 31 CFR 1010/1020), so I won't guess. Try rephrasing, or cite the "
    "section you have in mind."
)


@dataclass
class Citation:
    n: int
    chunk_id: str
    labels: list[str]
    cited_text: str
    url: str
    title: str


@dataclass
class Segment:
    text: str
    citations: list[int] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    answer: str  # plain text with [n] markers
    segments: list[Segment]
    citations: list[Citation]
    abstained: bool
    model: str | None
    usage: dict
    timings_ms: dict
    stop_reason: str | None = None

    @property
    def citation_coverage(self) -> float:
        """Share of answer characters that sit inside a cited segment."""
        total = sum(len(s.text.strip()) for s in self.segments)
        cited = sum(len(s.text.strip()) for s in self.segments if s.citations)
        return cited / total if total else 0.0


def build_search_results(hits: list[Hit]) -> list[dict]:
    return [
        {
            "type": "search_result",
            "source": h.chunk.url,
            "title": h.chunk.breadcrumb,
            "content": [{"type": "text", "text": f"[{b.label}] {b.text}"} for b in h.chunk.blocks],
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def parse_response(question: str, hits: list[Hit], content) -> tuple[list[Segment], list[Citation], str]:
    segments: list[Segment] = []
    citations: list[Citation] = []
    key_to_n: dict[tuple, int] = {}
    for block in content:
        if block.type != "text":
            continue  # thinking / fallback blocks
        seg = Segment(text=block.text)
        for c in getattr(block, "citations", None) or []:
            if getattr(c, "type", "") != "search_result_location":
                continue
            idx = c.search_result_index
            if not 0 <= idx < len(hits):
                continue  # defensive: never surface a citation we can't resolve
            chunk = hits[idx].chunk
            key = (chunk.id, c.start_block_index, c.end_block_index)
            if key not in key_to_n:
                key_to_n[key] = len(citations) + 1
                labels = [b.label for b in chunk.blocks[c.start_block_index : c.end_block_index]]
                citations.append(Citation(
                    n=key_to_n[key], chunk_id=chunk.id, labels=list(dict.fromkeys(labels)),
                    cited_text=c.cited_text, url=chunk.url, title=chunk.breadcrumb,
                ))
            if key_to_n[key] not in seg.citations:
                seg.citations.append(key_to_n[key])
        segments.append(seg)
    plain = "".join(s.text + "".join(f"[{n}]" for n in s.citations) for s in segments)
    return segments, citations, plain


class Generator:
    def __init__(self, settings: Settings | None = None, client: anthropic.Anthropic | None = None):
        self.settings = settings or get_settings()
        self.client = client or anthropic.Anthropic()

    def answer(self, question: str, retrieval: RetrievalResult) -> Answer:
        s = self.settings
        hits = retrieval.hits
        top = retrieval.top_score
        if not hits or (top is not None and top < s.abstain_threshold):
            return Answer(question, NO_ANSWER, [Segment(NO_ANSWER)], [], True, None, {},
                          dict(retrieval.timings_ms))

        t = time.perf_counter()
        with tracer.start_as_current_span("generate") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            set_io(span, question)
            response = self.client.beta.messages.create(
                model=s.anthropic_model,
                max_tokens=s.max_answer_tokens,
                system=SYSTEM_PROMPT,
                output_config={"effort": s.anthropic_effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{
                    "role": "user",
                    "content": [*build_search_results(hits), {"type": "text", "text": question}],
                }],
            )
            timings = dict(retrieval.timings_ms, llm_ms=(time.perf_counter() - t) * 1000)
            if response.stop_reason == "refusal":
                msg = "The model declined to answer this request."
                return Answer(question, msg, [Segment(msg)], [], True, response.model,
                              response.usage.to_dict(), timings, "refusal")
            segments, citations, plain = parse_response(question, hits, response.content)
            set_io(span, question, plain)
            return Answer(question, plain, segments, citations, False, response.model,
                          response.usage.to_dict(), timings, response.stop_reason)
