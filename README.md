# PipelineIQ

**An open-source multi-agent platform that automates the software development life cycle, powered by Gemma 4.**

Built at **Hacktoberfest Hack Day Milano × Coding Bunker** (17 October 2026). Entered in **Best Open-Source AI Project** and **Best Use of Gemma 4**.

## The problem

AI coding tools automate single tasks, but the life cycle around them stays manual. Requirements, design, code and tests live in different places, context is lost at every hand-off, and "done" depends on whoever reviews the work. It is hard to answer simple questions like *"which requirement does this test cover?"*

## The solution

PipelineIQ runs a small AI software team. You describe a product in one sentence (optionally with a whiteboard photo) and four specialised Gemma 4 agents take it through **Requirements → Design → Development → Testing**. After every phase an independent **Manager agent** reviews the work and decides:

| Decision | What happens |
|---|---|
| **APPROVE** | the next phase starts |
| **REWORK** | the work goes back with specific feedback, to the phase that caused the problem |
| **ESCALATE** | retry limit reached, a human takes over |

Every step is recorded, so you get full traceability from each requirement to the test that proves it.

## How it works

- **Phase agents** – Requirements, Design, Development and Testing, each with its own system prompt, output contract and knowledge base.
- **Shared state store** – Redux-style and event-sourced: agents never edit state directly, they dispatch actions (`REQUIREMENT_ADDED`, `TEST_RESULTS_RECORDED`, …). Every action is stored, so any project can be replayed and audited.
- **Manager agent** – runs **deterministic checks first** (code compiles, every design module exists, generated tests actually run with pytest and pass, every requirement has a test), then an **LLM review** against a rubric. When tests fail it does **root-cause routing**: it decides whether the code or the tests are wrong and sends the task to that agent.
- **Process log** – append-only record of every review: attempt, pass/fail, checks, score, decision, feedback.
- **Per-agent RAG** – each agent retrieves from its own collection (`knowledge/<agent>/`): team standards plus *lessons learned* from earlier Manager feedback, so the system stops repeating mistakes.
- **Live dashboard** – watch the pipeline, rework loops, RAG retrievals and Manager decisions in real time, and download the generated code as a zip.

## How Gemma 4 is used

- **Every agent runs on Gemma 4** through the Gemini API (`google-genai` SDK).
- **Model tiering:** `gemma-4-26b-a4b-it` (fast MoE) for the four worker agents, `gemma-4-31b-it` for the Manager's reviews.
- **Thinking mode** on for Manager reviews and root-cause analysis, minimal for fast drafting.
- **Multimodal input:** the Requirements agent reads whiteboard photos and UI sketches.
- **Structured JSON outputs** validated against a contract per agent; invalid replies are rejected and retried.
- **Resilience:** backoff on rate limits/server errors and automatic fallback to the other Gemma 4 model.

## How DigitalOcean is used

- **App Platform** hosts the FastAPI dashboard (`.do/app.yaml`).
- **Managed PostgreSQL** stores the action log, project state and process log.
- **pgvector** on the same database holds the per-agent RAG collections with an HNSW cosine index (`VECTOR_DB=pgvector`).

Locally everything runs on SQLite + ChromaDB with no setup.

## Tech stack

Python · Gemma 4 (Gemini API) · FastAPI · SQLAlchemy · PostgreSQL + pgvector / SQLite · ChromaDB · gemini-embedding-001 (or EmbeddingGemma, open weights, via `EMBEDDINGS=local`) · pytest · DigitalOcean App Platform

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your GEMINI_API_KEY
pytest                      # core tests (no API key needed)
python scripts/check_gemma.py
python scripts/dashboard.py # open http://localhost:8000
```

## Deploy on DigitalOcean

1. Create a Managed PostgreSQL cluster named `pipelineiq-db` (region `fra`).
2. Create an App from this repo and paste `.do/app.yaml` as the app spec.
3. Set `GEMINI_API_KEY` as a secret; add the app to the database's trusted sources.
4. Open `/api/health` — it should report `"vector_db": "pgvector"`.

> The testing phase executes AI-generated code. Run it in a throwaway container or VM.

## Demo

- **Live demo:** _add DigitalOcean URL_
- **Video:** _add link_

Demo scenario: *"Build a to-do API with OTP login"* → requirements → design → code → tests → the Manager sends at least one task back with specific feedback → approved, with every requirement traced to a passing test.

## What's next

Planning, Deployment and Maintenance agents, CI/CD and monitoring integrations, human approval gates in the dashboard, and a pilot on a real open-source repository.

## Team and original contributions

Everything in this repository was built during the Hack Day.

- Muhammad Qaseem – _build archietecture and features_
- Yui Watanabe – _improve the UI design_
- _Akash lanke_ – _ build features and ui_

## AI tools used

Declared per the Hack Day rules: Gemma 4 powers the product itself. During development the team used _list assistants, e.g. Claude, ChatGPT, Copilot_ for _what_.

## Third-party code

Libraries are used through their public packages (see `requirements.txt`); no external code was copied into the repository.

## License

MIT
