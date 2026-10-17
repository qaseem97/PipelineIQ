"""Central configuration, read from environment variables (or a .env file)."""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Gemma 4 models served through the Gemini API.
# Worker agents use the faster model, the Manager uses the larger one.
WORKER_MODEL = os.getenv("WORKER_MODEL", "gemma-4-26b-a4b-it")
MANAGER_MODEL = os.getenv("MANAGER_MODEL", "gemma-4-31b-it")

# SQLite works locally with zero setup. On DigitalOcean, point this at
# Managed PostgreSQL, e.g. postgresql+psycopg://user:pass@host:25060/db?sslmode=require
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///pipelineiq.db")

# Embeddings for the agents' RAG knowledge bases: gemini | local (EmbeddingGemma, open weights) | hash (offline)
EMBEDDINGS = os.getenv("EMBEDDINGS", "gemini")
# Vector database: chroma (local, ./chroma_db) | pgvector (PostgreSQL, e.g. DigitalOcean Managed PostgreSQL)
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", os.getenv("DATABASE_URL", ""))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# How many times the Manager may send a task back before escalating to a human.
MAX_REWORK_ATTEMPTS = int(os.getenv("MAX_REWORK_ATTEMPTS", "3"))

# Set USE_FAKE_LLM=1 to run everything offline with canned responses (tests, demos without a key).
USE_FAKE_LLM = os.getenv("USE_FAKE_LLM", "0") == "1"
