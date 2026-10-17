"""Vector databases for the agents' RAG knowledge bases.

    VECTOR_DB=chroma    ChromaDB, persisted on disk in ./chroma_db (local default, HNSW index)
    VECTOR_DB=pgvector  PostgreSQL + pgvector (DigitalOcean Managed PostgreSQL in production)

Both store one collection per agent. Collections are namespaced by the embedder
(e.g. "testing__gemini768") so vectors from different embedding models never mix.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from . import config

log = logging.getLogger("pipelineiq.vectorstore")


class VectorStore(Protocol):
    def existing_ids(self, collection: str) -> set[str]: ...
    def add(self, collection: str, ids: list[str], texts: list[str], embeddings: list[list[float]],
            sources: list[str]) -> None: ...
    def query(self, collection: str, embedding: list[float], k: int) -> list[dict]: ...
    def counts(self) -> dict[str, int]: ...
    def clear(self, collection: str | None = None) -> None: ...


# ---------------------------------------------------------------------------
# ChromaDB (local)
# ---------------------------------------------------------------------------
class ChromaVectorStore:
    def __init__(self, path: str | None = None, namespace: str = "default"):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(path=path or config.CHROMA_PATH,
                                                settings=Settings(anonymized_telemetry=False))
        self.namespace = namespace

    def _name(self, collection: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", f"{collection}__{self.namespace}")[:63]

    def _col(self, collection: str):
        return self.client.get_or_create_collection(
            self._name(collection), embedding_function=None,
            metadata={"hnsw:space": "cosine", "agent": collection})

    def existing_ids(self, collection):
        return set(self._col(collection).get(include=[])["ids"])

    def add(self, collection, ids, texts, embeddings, sources):
        self._col(collection).upsert(ids=ids, documents=texts, embeddings=embeddings,
                                     metadatas=[{"source": s} for s in sources])

    def query(self, collection, embedding, k):
        col = self._col(collection)
        n = col.count()
        if n == 0:
            return []
        try:
            res = col.query(query_embeddings=[embedding], n_results=min(k, n),
                            include=["documents", "metadatas", "distances"])
            return [{"text": d, "source": m["source"], "score": round(1 - dist, 3)}
                    for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
        except Exception as first:  # noqa: BLE001
            # Chroma sometimes fails to open a freshly written HNSW index (seen on Windows:
            # "Error creating hnsw segment reader: Nothing found on disk"). Try an exact
            # cosine search over the stored records; if Chroma cannot read those either,
            # carry on without retrieved context rather than stopping the pipeline.
            try:
                return self._exact_search(col, embedding, k)
            except Exception as second:  # noqa: BLE001
                log.warning("Knowledge search skipped for %s: %s / %s", collection, first, second)
                return []

    @staticmethod
    def _exact_search(col, embedding, k):
        data = col.get(include=["documents", "metadatas", "embeddings"])
        q = [float(x) for x in embedding]
        qn = sum(x * x for x in q) ** 0.5 or 1.0
        scored = []
        for doc, meta, emb in zip(data["documents"], data["metadatas"], data["embeddings"]):
            e = [float(x) for x in emb]
            en = sum(x * x for x in e) ** 0.5 or 1.0
            scored.append((sum(a * b for a, b in zip(q, e)) / (qn * en), doc, meta))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [{"text": d, "source": m["source"], "score": round(s, 3)} for s, d, m in scored[:k]]

    def counts(self):
        out = {}
        suffix = f"__{self.namespace}"
        for col in self.client.list_collections():
            name = col if isinstance(col, str) else col.name
            if name.endswith(suffix):
                out[name[: -len(suffix)]] = self.client.get_collection(name, embedding_function=None).count()
        return out

    def clear(self, collection=None):
        names = [self._name(collection)] if collection else [
            (c if isinstance(c, str) else c.name) for c in self.client.list_collections()]
        for n in names:
            try:
                self.client.delete_collection(n)
            except Exception:  # noqa: BLE001 - already gone
                pass


# ---------------------------------------------------------------------------
# PostgreSQL + pgvector (DigitalOcean)
# ---------------------------------------------------------------------------
class PgVectorStore:
    """Uses one table per embedder namespace with an HNSW cosine index."""

    def __init__(self, url: str | None = None, namespace: str = "default"):
        import psycopg

        url = url or config.VECTOR_DB_URL
        # SQLAlchemy-style URLs (postgresql+psycopg://) -> plain libpq URL
        self.url = re.sub(r"^postgresql\+\w+://", "postgresql://", url)
        self.psycopg = psycopg
        self.table = re.sub(r"[^a-z0-9_]", "_", f"kb_vectors_{namespace}".lower())
        self._ready = False

    def _conn(self):
        return self.psycopg.connect(self.url, autocommit=True)

    def _ensure(self, dim: int | None = None):
        if self._ready:
            return True
        with self._conn() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            exists = c.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0]
            if not exists:
                if dim is None:
                    return False
                c.execute(f"""CREATE TABLE {self.table} (
                    collection text NOT NULL, id text NOT NULL, source text NOT NULL,
                    text text NOT NULL, embedding vector({dim}) NOT NULL,
                    PRIMARY KEY (collection, id))""")
                c.execute(f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops)")
        self._ready = True
        return True

    @staticmethod
    def _vec(v):
        return "[" + ",".join(f"{x:.7f}" for x in v) + "]"

    def existing_ids(self, collection):
        if not self._ensure():
            return set()
        with self._conn() as c:
            return {r[0] for r in c.execute(f"SELECT id FROM {self.table} WHERE collection=%s", (collection,))}

    def add(self, collection, ids, texts, embeddings, sources):
        self._ensure(len(embeddings[0]))
        with self._conn() as c, c.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table} (collection, id, source, text, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (collection, id) DO NOTHING""",
                [(collection, i, s, t, self._vec(e)) for i, t, e, s in zip(ids, texts, embeddings, sources)])

    def query(self, collection, embedding, k):
        if not self._ensure():
            return []
        with self._conn() as c:
            rows = c.execute(
                f"""SELECT text, source, 1 - (embedding <=> %s::vector) AS score FROM {self.table}
                    WHERE collection=%s ORDER BY embedding <=> %s::vector LIMIT %s""",
                (self._vec(embedding), collection, self._vec(embedding), k)).fetchall()
        return [{"text": t, "source": s, "score": round(float(sc), 3)} for t, s, sc in rows]

    def counts(self):
        if not self._ensure():
            return {}
        with self._conn() as c:
            return {r[0]: r[1] for r in c.execute(
                f"SELECT collection, count(*) FROM {self.table} GROUP BY collection")}

    def clear(self, collection=None):
        if not self._ensure():
            return
        with self._conn() as c:
            if collection:
                c.execute(f"DELETE FROM {self.table} WHERE collection=%s", (collection,))
            else:
                c.execute(f"DELETE FROM {self.table}")


def get_vector_store(namespace: str) -> VectorStore:
    if config.VECTOR_DB == "pgvector":
        return PgVectorStore(namespace=namespace)
    return ChromaVectorStore(namespace=namespace)


__all__ = ["ChromaVectorStore", "PgVectorStore", "VectorStore", "get_vector_store"]
