"""Structure-aware chunking for regulatory text.

Rules (see README "Chunking strategy"):
  * A chunk never crosses a section boundary, and is built from whole paragraphs.
  * Paragraphs are grouped by top-level paragraph, e.g. § 1026.54(a), or by
    Official-Interpretation heading, e.g. Comment 54(a)(1).
  * Groups larger than ``max_words`` are split at paragraph boundaries; small
    adjacent groups in the same section are merged up to ``max_words``.
  * Tables are kept whole as their own block (rendered as Markdown).
  * Every chunk carries a breadcrumb header (regulation › section › paragraph)
    that is prepended for embedding and BM25, so a chunk like "(ii) Any portion
    of a balance…" is still findable by "grace period Regulation Z".
  * Each paragraph is a separate *block*: the LLM cites blocks, so citations
    resolve to an exact paragraph label rather than a whole chunk.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

from .ecfr import SOURCES, Paragraph

MAX_WORDS = 300
MIN_WORDS = 80

LONG_NAMES = {doc: long for _, _, doc, long in SOURCES}
_XREF = re.compile(r"§§?\s*(\d{3,4}\.\d+)")


@dataclass
class Block:
    label: str
    text: str


@dataclass
class Chunk:
    id: str
    doc: str
    section_id: str
    kind: str
    title: str
    breadcrumb: str
    url: str
    blocks: list[Block]
    cross_refs: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    @property
    def search_text(self) -> str:
        """Text used for embedding and BM25: breadcrumb header + body."""
        return f"{self.breadcrumb}\n{self.body}"

    @property
    def labels(self) -> list[str]:
        return [b.label for b in self.blocks]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        d = dict(d)
        d["blocks"] = [Block(**b) for b in d["blocks"]]
        return cls(**d)


def _words(p: Paragraph) -> int:
    return len(p.text.split())


def _range_label(labels: list[str]) -> str:
    first, last = labels[0], labels[-1]
    return first if first == last else f"{first} – {last.replace('§ ', '')}"


def _make_chunk(paras: list[Paragraph], seq: int) -> Chunk:
    p0 = paras[0]
    labels = list(dict.fromkeys(p.label for p in paras))
    span = _range_label(labels)
    title = p0.section_title
    long = LONG_NAMES.get(p0.doc, "")
    crumbs = [f"{p0.doc} ({long})" if long else p0.doc, *p0.heading_path[1:], span]
    breadcrumb = " › ".join(dict.fromkeys(crumbs))
    body = "\n".join(p.text for p in paras)
    refs = sorted({r for r in _XREF.findall(body) if r != p0.section_id})
    digest = hashlib.sha1(f"{seq}|{p0.doc}|{span}|{body}".encode()).hexdigest()[:12]
    return Chunk(
        id=digest, doc=p0.doc, section_id=p0.section_id, kind=p0.kind, title=title,
        breadcrumb=breadcrumb, url=paras[0].url,
        blocks=[Block(label=p.label, text=p.text) for p in paras], cross_refs=refs,
    )


def _split_group(paras: list[Paragraph], max_words: int) -> list[list[Paragraph]]:
    pieces: list[list[Paragraph]] = []
    cur: list[Paragraph] = []
    n = 0
    for p in paras:
        w = _words(p)
        if p.is_table:
            if cur:
                pieces.append(cur)
            pieces.append([p])
            cur, n = [], 0
            continue
        if cur and n + w > max_words:
            pieces.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        pieces.append(cur)
    return pieces


def chunk_paragraphs(
    paragraphs: list[Paragraph], max_words: int = MAX_WORDS, min_words: int = MIN_WORDS
) -> list[Chunk]:
    # 1. consecutive paragraphs sharing (doc, group) form a group
    groups: list[list[Paragraph]] = []
    for p in paragraphs:
        if groups and groups[-1][0].doc == p.doc and groups[-1][0].group == p.group:
            groups[-1].append(p)
        else:
            groups.append([p])

    # 2. split oversized groups at paragraph boundaries
    pieces: list[list[Paragraph]] = []
    for g in groups:
        pieces.extend(_split_group(g, max_words))

    # 3. merge undersized neighbours within the same section
    merged: list[list[Paragraph]] = []
    for piece in pieces:
        if merged:
            prev = merged[-1]
            same_section = prev[0].doc == piece[0].doc and prev[0].section_id == piece[0].section_id
            prev_words = sum(_words(p) for p in prev)
            piece_words = sum(_words(p) for p in piece)
            no_tables = not prev[-1].is_table and not piece[0].is_table
            if (
                same_section and no_tables
                and (prev_words < min_words or piece_words < min_words)
                and prev_words + piece_words <= max_words
            ):
                prev.extend(piece)
                continue
        merged.append(list(piece))

    chunks = [_make_chunk(m, i) for i, m in enumerate(merged)]
    # Drop pure boilerplate (e.g. "[Reserved]") chunks.
    return [c for c in chunks if len(c.body.split()) >= 3]
