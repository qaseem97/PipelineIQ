"""Per-agent RAG knowledge bases.

Each agent has its own collection (named after the agent) in a real vector database
(ChromaDB locally, PostgreSQL + pgvector on DigitalOcean — see vectorstore.py).
A collection holds:
  * curated guidance from knowledge/<agent>/*.md  (team standards, best practices)
  * lessons learned: every piece of Manager feedback from earlier projects, so agents
    stop repeating the same mistakes (the system improves with use).

Embedders (set EMBEDDINGS in .env):
  gemini  - gemini-embedding-001 through the Gemini API (default, same API key)
  local   - EmbeddingGemma (open weights) via sentence-transformers, fully offline
  hash    - dependency-free hashed bag-of-words, for tests and offline demos
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from . import config
from .vectorstore import VectorStore, get_vector_store


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Deterministic, offline embedding: hashed word and word-pair counts."""
    dim = 512
    id = "hash512"

    def _one(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self.dim
        for tok in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        return _normalize(vec)

    def embed_documents(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)


class GeminiEmbedder:
    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.types, self.model, self.dim = types, model, dim
        self.id = f"gemini{dim}"

    def _embed(self, texts, task):
        out = []
        for i in range(0, len(texts), 50):
            res = self.client.models.embed_content(
                model=self.model, contents=texts[i:i + 50],
                config=self.types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim))
            out += [_normalize(list(e.values)) for e in res.embeddings]
        return out

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class LocalGemmaEmbedder:
    """EmbeddingGemma, open weights. Needs: pip install sentence-transformers (and a Hugging Face
    login with the model licence accepted)."""

    def __init__(self, model: str = "google/embeddinggemma-300m"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)
        self.id = "embeddinggemma"

    def embed_documents(self, texts):
        return [_normalize(list(map(float, v))) for v in self.m.encode(list(texts))]

    def embed_query(self, text):
        return _normalize(list(map(float, self.m.encode([text])[0])))


def get_embedder():
    kind = config.EMBEDDINGS
    if kind == "hash" or config.USE_FAKE_LLM:
        return HashEmbedder()
    if kind == "local":
        return LocalGemmaEmbedder()
    return GeminiEmbedder()


def chunk_markdown(text: str, max_chars: int = 900) -> list[str]:
    """Split on headings, then pack paragraphs into chunks of up to max_chars,
    keeping the heading with each chunk so it stays meaningful on its own."""
    chunks = []
    for section in re.split(r"\n(?=#{1,3} )", text.strip()):
        lines = section.strip().splitlines()
        if not lines:
            continue
        heading = lines[0] if lines[0].startswith("#") else ""
        body = "\n".join(lines[1:] if heading else lines).strip()
        buf = ""
        for para in re.split(r"\n\s*\n", body):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append(f"{heading}\n{buf}".strip())
                buf = ""
            buf = f"{buf}\n\n{para}".strip()
        if buf or heading:
            chunks.append(f"{heading}\n{buf}".strip())
    return [c for c in chunks if len(c) > 20]


class KnowledgeBase:
    def __init__(self, embedder=None, vector_store: VectorStore | None = None):
        self._embedder = embedder
        self._store = vector_store

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = get_vector_store(namespace=self.embedder.id)
        return self._store

    def add_texts(self, collection: str, source: str, texts: list[str]) -> int:
        """Embed and store texts; skips any text already stored in this collection."""
        existing = self.store.existing_ids(collection)
        new = [(hashlib.sha256(t.encode()).hexdigest()[:32], t) for t in texts]
        new = [(d, t) for d, t in dict(new).items() if d not in existing]
        if not new:
            return 0
        vectors = self.embedder.embed_documents([t for _, t in new])
        self.store.add(collection, [d for d, _ in new], [t for _, t in new], vectors, [source] * len(new))
        return len(new)

    def ingest_dir(self, root: str | Path = "knowledge") -> dict[str, int]:
        """knowledge/<collection>/*.md -> chunks in <collection>."""
        added = {}
        for folder in sorted(Path(root).iterdir()) if Path(root).exists() else []:
            if folder.is_dir():
                for f in sorted(folder.glob("*.md")):
                    n = self.add_texts(folder.name, f"{folder.name}/{f.name}",
                                       chunk_markdown(f.read_text(encoding="utf-8")))
                    added[folder.name] = added.get(folder.name, 0) + n
        return added

    def add_lesson(self, collection: str, project_idea: str, feedback: str) -> None:
        text = f"# Lesson from a past review\nProject: {project_idea}\nReviewer feedback that required rework:\n{feedback}"
        self.add_texts(collection, "lessons-learned", [text])

    def search(self, collection: str, query: str, k: int = 4, min_score: float = 0.05) -> list[dict]:
        hits = self.store.query(collection, self.embedder.embed_query(query), k)
        return [h for h in hits if h["score"] >= min_score]

    def stats(self) -> dict[str, int]:
        return self.store.counts()

    def clear(self, collection: str | None = None) -> None:
        self.store.clear(collection)
