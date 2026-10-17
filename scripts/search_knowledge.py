"""Search an agent's knowledge base, e.g.:
    python scripts/search_knowledge.py testing "how do I test an error is raised"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelineiq.orchestrator import KNOWLEDGE_DIR
from pipelineiq.rag import KnowledgeBase

collection, query = sys.argv[1], " ".join(sys.argv[2:])
kb = KnowledgeBase()
kb.ingest_dir(KNOWLEDGE_DIR)
print(f"Collections: {kb.stats()}\n")
for hit in kb.search(collection, query):
    print(f"--- {hit['source']}  (score {hit['score']})\n{hit['text']}\n")
