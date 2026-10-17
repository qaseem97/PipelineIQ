import json

from pipelineiq.agents import AGENTS
from pipelineiq.db import get_engine
from pipelineiq.llm import FakeLLM
from pipelineiq.orchestrator import KNOWLEDGE_DIR
from pipelineiq.rag import HashEmbedder, KnowledgeBase, chunk_markdown
from pipelineiq.vectorstore import ChromaVectorStore
from pipelineiq.store import Store


def kb_for(tmp_path):
    from pipelineiq import config
    return KnowledgeBase(embedder=HashEmbedder(),
                         vector_store=ChromaVectorStore(path=config.CHROMA_PATH, namespace=f"kb_{tmp_path.name}"))


def test_chunking_keeps_headings():
    chunks = chunk_markdown("# Alpha\nalpha guidance text here\n\n# Beta\nbeta guidance text here")
    assert chunks == ["# Alpha\nalpha guidance text here", "# Beta\nbeta guidance text here"]


def test_ingest_is_idempotent_and_per_collection(tmp_path):
    kb = kb_for(tmp_path)
    first = kb.ingest_dir(KNOWLEDGE_DIR)
    assert set(first) == {"requirements", "design", "development", "testing", "manager"}
    assert kb.ingest_dir(KNOWLEDGE_DIR) == {k: 0 for k in first}   # nothing re-embedded


def test_search_returns_relevant_chunk(tmp_path):
    kb = kb_for(tmp_path)
    kb.ingest_dir(KNOWLEDGE_DIR)
    hits = kb.search("testing", "how to test that an error is raised with pytest raises")
    assert hits and "pytest.raises" in hits[0]["text"]
    assert all("testing/" in h["source"] for h in hits)


def test_lessons_are_retrieved_later(tmp_path):
    kb = kb_for(tmp_path)
    kb.add_lesson("development", "library system", "Use absolute imports like from app.models import Book")
    hits = kb.search("development", "library lending system absolute imports")
    assert hits[0]["source"] == "lessons-learned"


def test_agent_prompt_includes_retrieved_knowledge(tmp_path):
    kb = kb_for(tmp_path)
    kb.ingest_dir(KNOWLEDGE_DIR)
    store = Store.create("Todo", "to-do list with priorities", engine=get_engine(f"sqlite:///{tmp_path / 's.db'}"))
    store.dispatch("TASK_CREATED", "orchestrator", {"task_id": "R1", "phase": "requirements", "agent": "requirements"})
    reply = {"items": [{"id": "REQ-1", "title": "t", "description": "d", "acceptance_criteria": ["c"]}]}
    llm = FakeLLM(lambda p, s: json.dumps(reply))
    agent = AGENTS["requirements"](store, llm, kb=kb)
    agent.run("R1")
    assert "Retrieved knowledge" in llm.calls[0]["prompt"]
    assert agent.retrieved and agent.retrieved[0]["source"].startswith("requirements/")


def test_knowledge_persists_in_vector_db(tmp_path):
    kb = kb_for(tmp_path)
    kb.ingest_dir(KNOWLEDGE_DIR)
    reopened = kb_for(tmp_path)                       # new client, same folder on disk
    assert reopened.stats() == kb.stats() and sum(kb.stats().values()) > 10
    assert reopened.search("design", "how should ids be generated")[0]["source"] == "design/design_guidelines.md"


def test_search_falls_back_when_chroma_index_fails(tmp_path, monkeypatch):
    """Simulates Chroma's 'Error creating hnsw segment reader: Nothing found on disk'."""
    kb = kb_for(tmp_path)
    kb.ingest_dir(KNOWLEDGE_DIR)
    import chromadb.api.models.Collection as C

    def broken(self, *a, **k):
        raise RuntimeError("Error creating hnsw segment reader: Nothing found on disk")
    monkeypatch.setattr(C.Collection, "query", broken)
    hits = kb.search("testing", "how to test that an error is raised with pytest raises")
    assert hits and "pytest.raises" in hits[0]["text"]


def test_search_degrades_gracefully_when_chroma_cannot_read(tmp_path, monkeypatch):
    kb = kb_for(tmp_path)
    kb.ingest_dir(KNOWLEDGE_DIR)
    import chromadb.api.models.Collection as C

    def broken(self, *a, **k):
        raise RuntimeError("Error creating hnsw segment reader: Nothing found on disk")
    monkeypatch.setattr(C.Collection, "query", broken)
    monkeypatch.setattr(C.Collection, "get", broken)
    assert kb.search("testing", "pytest raises") == []   # no crash; pipeline continues
