"""Ingest the eCFR corpus: download → parse → chunk → embed → Postgres.

    python -m regrag.ingest [--reset]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .chunking import chunk_paragraphs
from .config import get_settings
from .ecfr import load_corpus
from .retrieval import Embedder
from .store import Store

log = logging.getLogger("regrag.ingest")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop and rebuild the chunks table")
    args = parser.parse_args()
    s = get_settings()

    t0 = time.perf_counter()
    paragraphs = load_corpus(s.data_dir, s.ecfr_date)
    chunks = chunk_paragraphs(paragraphs)
    log.info("Parsed %d paragraphs into %d chunks", len(paragraphs), len(chunks))

    # Chunk manifest: used by the golden-set builder and for reproducibility.
    manifest = Path(s.data_dir) / "chunks.jsonl"
    with manifest.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")

    store = Store(s.database_url, s.embed_dim)
    store.init_schema(reset=args.reset)
    # Resumable: chunk IDs are deterministic, so only embed what's missing.
    have = store.ids()
    stale = have - {c.id for c in chunks}
    if stale:
        store.delete(stale)
    todo = [(i, c) for i, c in enumerate(chunks) if c.id not in have]
    if not todo:
        log.info("Store already holds all %d chunks; nothing to do", len(chunks))
        return

    embedder = Embedder(s.embed_model)
    batch = 128
    for j in range(0, len(todo), batch):
        part = todo[j : j + batch]
        vecs = embedder.embed_passages([c.search_text for _, c in part])
        for (seq, c), v in zip(part, vecs, strict=False):
            store.upsert([c], [v], start_seq=seq)
        log.info("Embedded %d/%d", min(j + batch, len(todo)), len(todo))
    log.info("Done: %d chunks in %.1fs", store.count(), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
