import pytest

from pipelineiq import config, rag
from pipelineiq.vectorstore import ChromaVectorStore


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    """Tests never call the embedding API."""
    monkeypatch.setattr(config, "EMBEDDINGS", "hash")


@pytest.fixture(scope="session")
def chroma_dir(tmp_path_factory):
    # ChromaDB's engine binds to the first folder opened in a process, so all tests
    # share one folder and stay isolated through unique collection names instead.
    return str(tmp_path_factory.mktemp("chroma"))


@pytest.fixture(autouse=True)
def temp_vector_db(chroma_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_DB", "chroma")
    monkeypatch.setattr(config, "CHROMA_PATH", chroma_dir)
    monkeypatch.setattr(rag, "get_vector_store",
                        lambda namespace: ChromaVectorStore(path=chroma_dir, namespace=f"{namespace}_{tmp_path.name}"))
