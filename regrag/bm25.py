"""BM25 keyword index with a tokenizer that understands CFR citations.

Generic tokenizers split "§ 1026.54(a)(1)" into "1026", "54", "a", "1" — which
matches every section in Part 1026. We emit the citation itself at several
granularities ("1026.54", "1026.54(a)", "1026.54(a)(1)") and normalise
interpretation references ("comment 54(a)(1)-1" → "c54(a)(1)-1"), plus common
aliases ("reg z" → "regulation_z", "tila"), so exact-citation queries hit.
"""

from __future__ import annotations

import re

import bm25s
import numpy as np
import Stemmer

_STOP = set(
    "a an and are as at be by for from has have if in into is it its of on or such that the "
    "their then there these this to was were which will with may must shall any under".split()
)
_CITE = re.compile(r"(\d{3,4}\.\d+)((?:\([a-zA-Z0-9]+\))*)")
_COMMENT = re.compile(r"comment\s+(\d+(?:\([a-zA-Z0-9]+\))*-\d+)", re.I)
_ALIASES = [
    (re.compile(r"\breg(?:ulation)?\.?\s+z\b", re.I), " regulation_z truth_in_lending "),
    (re.compile(r"\breg(?:ulation)?\.?\s+e\b", re.I), " regulation_e electronic_fund_transfer "),
    (re.compile(r"\btila\b|\btruth in lending\b", re.I), " truth_in_lending "),
    (re.compile(r"\befta\b|\belectronic fund transfers?\b", re.I), " electronic_fund_transfer "),
    (re.compile(r"\bbsa\b|\bbank secrecy act\b", re.I), " bank_secrecy_act "),
    (re.compile(r"\bcip\b|\bcustomer identification program\b", re.I), " customer_identification_program "),
]

_stemmer = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _CITE.finditer(text):
        base, subs = m.group(1), m.group(2)
        tokens.append(base)
        acc = base
        for s in re.findall(r"\([a-zA-Z0-9]+\)", subs):
            acc += s.lower()
            tokens.append(acc)
    for m in _COMMENT.finditer(text):
        tokens.append("c" + m.group(1).lower())
    for pat, repl in _ALIASES:
        text = pat.sub(repl, text)
    words = [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOP]
    tokens.extend(_stemmer.stemWords(words))
    return tokens


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]) -> None:
        self.ids = ids
        self.retriever = bm25s.BM25(k1=1.2, b=0.75)
        self.retriever.index([tokenize(t) for t in texts], show_progress=False)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q:
            return []
        k = min(k, len(self.ids))
        docs, scores = self.retriever.retrieve([q], k=k, show_progress=False)
        out = []
        for idx, score in zip(np.asarray(docs[0]), np.asarray(scores[0]), strict=False):
            if score > 0:
                out.append((self.ids[int(idx)], float(score)))
        return out
