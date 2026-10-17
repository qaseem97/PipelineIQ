# PipelineIQ

**An open-source multi-agent platform that automates the software development life cycle, powered by Gemma 4.**

Built for Hacktoberfest 2026 (MLH Hack Day) — entered in *Best Open-Source AI Project* (DigitalOcean) and *Best Use of Gemma 4*.

## How it works

- **Phase agents** — Requirements, Design, Development and Testing agents, each a specialised Gemma 4 agent.
- **Shared state store** — Redux-style: agents dispatch actions, reducers update one shared project state. Every action is saved, so the state can be replayed and audited.
- **Manager agent** — reviews every submission and decides APPROVE, REWORK or ESCALATE.
- **Process log** — append-only record of every review (pass/fail, checks, feedback).
- **Per-agent RAG** — every agent retrieves from its own knowledge base (`knowledge/<agent>/`): team standards plus *lessons learned* from past Manager feedback, so the system stops repeating mistakes. Vectors live in a real vector database: **ChromaDB** locally (HNSW index) or **PostgreSQL + pgvector** on DigitalOcean (`VECTOR_DB=pgvector`).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env  — then add your GEMINI_API_KEY
pytest                      # core tests (no API key needed)
python scripts/check_gemma.py
```

## Live dashboard

```bash
python scripts/dashboard.py      # then open http://localhost:8000
```

Describe a product, optionally attach a whiteboard photo, and watch the pipeline: the assembly line shows each phase and every rework loop the Manager triggers, the activity feed shows RAG retrieval and review decisions as they happen, and tabs show the requirements, design, code, tests, process log and every state action. Generated code can be downloaded as a zip.

## Tech

Gemma 4 via the Gemini API · Python · SQLAlchemy (SQLite locally, DigitalOcean Managed PostgreSQL in production) · DigitalOcean App Platform

## License

MIT
