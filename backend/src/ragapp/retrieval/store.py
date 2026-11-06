import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import faiss
import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ragapp.config import settings
from ragapp.models import Chunk, Document

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
        )
    return _pool


@contextmanager
def get_connection():
    with _get_pool().connection() as conn:
        yield conn


def init_schema() -> None:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    try:
        conn.execute(SCHEMA_PATH.read_text())
    finally:
        conn.close()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VectorStore:
    """Hybrid store: Postgres for documents + query logs, FAISS for vector search.

    Chunk embeddings and metadata live entirely in FAISS — no chunks table in Postgres.
    The FAISS index and a companion meta.json file are persisted to faiss_index_path.
    """

    def __init__(self, index_dir: str | None = None):
        self._dir = Path(index_dir or settings.faiss_index_path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._dir / "index.bin"
        self._meta_file = self._dir / "meta.json"
        self._meta: list[dict] = []
        self._index: faiss.Index | None = None
        self._load()

    # --- FAISS persistence ---

    def _load(self) -> None:
        if self._meta_file.exists():
            with open(self._meta_file) as f:
                self._meta = json.load(f)
        if self._index_file.exists() and self._meta:
            self._index = faiss.read_index(str(self._index_file))

    def _save(self) -> None:
        with open(self._meta_file, "w") as f:
            json.dump(self._meta, f)
        if self._index is not None and self._index.ntotal > 0:
            faiss.write_index(self._index, str(self._index_file))
        elif self._index_file.exists():
            self._index_file.unlink()

    def _rebuild_index(self) -> None:
        if not self._meta:
            self._index = None
            return
        dim = len(self._meta[0]["embedding"])
        embeddings = np.array([m["embedding"] for m in self._meta], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index

    # --- Document store (Postgres) ---

    def upsert_document(self, filename: str, title: str, full_text_hash: str) -> str:
        with get_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO documents (filename, title, content_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (content_hash) DO UPDATE SET title = EXCLUDED.title
                RETURNING id
                """,
                (filename, title, full_text_hash),
            ).fetchone()
            return str(row[0])

    def upsert_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT filename, title FROM documents WHERE id = %s", (document_id,)
            ).fetchone()
        filename, title = (row[0], row[1]) if row else ("", "")

        self._meta = [m for m in self._meta if m["document_id"] != document_id]
        for chunk in chunks:
            chunk_id = hashlib.sha256(f"{document_id}:{chunk.chunk_index}".encode()).hexdigest()[:32]
            self._meta.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "title": title,
                "content": chunk.content,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "embedding": chunk.embedding,
            })
        self._rebuild_index()
        self._save()

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, Document]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        # Over-fetch when filtering so we still get top_k after the filter
        search_k = min(self._index.ntotal, top_k * 10 if document_ids else top_k)
        _, indices = self._index.search(q, search_k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[idx]
            if document_ids and m["document_id"] not in document_ids:
                continue
            results.append((
                Chunk(
                    id=m["chunk_id"],
                    content=m["content"],
                    page_number=m["page_number"],
                    chunk_index=m["chunk_index"],
                    document_id=m["document_id"],
                ),
                Document(id=m["document_id"], filename=m["filename"], title=m["title"]),
            ))
            if len(results) >= top_k:
                break
        return results

    def list_documents(self) -> list[Document]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, filename, title FROM documents ORDER BY ingested_at"
            ).fetchall()
        return [Document(id=str(r[0]), filename=r[1], title=r[2]) for r in rows]

    # --- Query logging (Postgres) ---

    def log_query(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[dict],
        cited_sources: list[dict],
        refused: bool,
        retrieval_latency_ms: int,
        generation_latency_ms: int,
        embed_latency_ms: int | None = None,
        search_latency_ms: int | None = None,
    ) -> str:
        with get_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO query_logs
                    (question, answer, retrieved_chunks, cited_sources, refused,
                     retrieval_latency_ms, embed_latency_ms, search_latency_ms, generation_latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    question,
                    answer,
                    Jsonb(retrieved_chunks),
                    Jsonb(cited_sources),
                    refused,
                    retrieval_latency_ms,
                    embed_latency_ms,
                    search_latency_ms,
                    generation_latency_ms,
                ),
            ).fetchone()
            return str(row[0])

    def set_feedback(self, query_log_id: str, feedback: int) -> None:
        with get_connection() as conn:
            conn.execute(
                "UPDATE query_logs SET feedback = %s WHERE id = %s", (feedback, query_log_id)
            )

    def list_query_logs(self, limit: int = 20) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, question, answer, retrieved_chunks, cited_sources, refused,
                       retrieval_latency_ms, embed_latency_ms, search_latency_ms,
                       generation_latency_ms, feedback, created_at
                FROM query_logs
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        columns = [
            "id", "question", "answer", "retrieved_chunks", "cited_sources", "refused",
            "retrieval_latency_ms", "embed_latency_ms", "search_latency_ms",
            "generation_latency_ms", "feedback", "created_at",
        ]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def get_stats(self) -> dict:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    count(*) AS total_queries,
                    avg(retrieval_latency_ms) AS avg_retrieval_latency_ms,
                    avg(generation_latency_ms) AS avg_generation_latency_ms,
                    avg(case when refused then 1.0 else 0.0 end) AS refusal_rate,
                    count(*) FILTER (WHERE feedback = 1) AS thumbs_up,
                    count(*) FILTER (WHERE feedback = -1) AS thumbs_down,
                    count(*) FILTER (WHERE feedback IS NOT NULL) AS feedback_count
                FROM query_logs
                """
            ).fetchone()
        return {
            "total_queries": row[0],
            "avg_retrieval_latency_ms": round(row[1]) if row[1] is not None else None,
            "avg_generation_latency_ms": round(row[2]) if row[2] is not None else None,
            "refusal_rate": round(row[3], 3) if row[3] is not None else None,
            "thumbs_up": row[4],
            "thumbs_down": row[5],
            "feedback_count": row[6],
        }
