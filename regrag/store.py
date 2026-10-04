"""Postgres + pgvector storage for chunks and their dense embeddings."""

from __future__ import annotations

import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .chunking import Chunk

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    seq         INTEGER NOT NULL,
    doc         TEXT NOT NULL,
    section_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    breadcrumb  TEXT NOT NULL,
    url         TEXT NOT NULL,
    blocks      JSONB NOT NULL,
    cross_refs  TEXT[] NOT NULL DEFAULT '{}',
    search_text TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc);
"""


class Store:
    def __init__(self, database_url: str, dim: int) -> None:
        self.conn = psycopg.connect(database_url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.dim = dim

    def init_schema(self, reset: bool = False) -> None:
        if reset:
            self.conn.execute("DROP TABLE IF EXISTS chunks")
        self.conn.execute(SCHEMA.replace("{dim}", str(self.dim)))

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray, start_seq: int = 0) -> None:
        rows = [
            (
                c.id, start_seq + i, c.doc, c.section_id, c.kind, c.title, c.breadcrumb, c.url,
                json.dumps([b.__dict__ for b in c.blocks]), c.cross_refs, c.search_text, emb,
            )
            for i, (c, emb) in enumerate(zip(chunks, embeddings, strict=False))
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (id, seq, doc, section_id, kind, title, breadcrumb, url,
                                       blocks, cross_refs, search_text, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding""",
                rows,
            )

    def ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT id FROM chunks").fetchall()}

    def delete(self, ids: set[str]) -> None:
        self.conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (list(ids),))

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            """SELECT id, doc, section_id, kind, title, breadcrumb, url, blocks, cross_refs
               FROM chunks ORDER BY seq"""
        ).fetchall()
        return [
            Chunk.from_dict(dict(
                id=r[0], doc=r[1], section_id=r[2], kind=r[3], title=r[4], breadcrumb=r[5],
                url=r[6], blocks=r[7], cross_refs=list(r[8]),
            ))
            for r in rows
        ]

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        # ef_search must be >= k for HNSW to return k rows.
        self.conn.execute(f"SET hnsw.ef_search = {max(40, k * 2)}")
        rows = self.conn.execute(
            "SELECT id, 1 - (embedding <=> %s) AS sim FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, query_vec, k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]
