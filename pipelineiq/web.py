"""Live dashboard: start projects and watch the agents, Manager, RAG and process log in real time.

    python scripts/dashboard.py          -> http://localhost:8000
"""
from __future__ import annotations

import io
import threading
import time
import traceback
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from . import config
from .agents import PHASE_ORDER
from .llm import TrackedLLM, get_llm
from .orchestrator import Orchestrator
from .process_log import ProcessLog
from .rag import KnowledgeBase
from .store import Store, list_projects

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="PipelineIQ")

_lock = threading.Lock()
_running: dict[str, Store] = {}     # project_id -> live store while its pipeline runs
_events: dict[str, list[dict]] = {}  # project_id -> live event feed
_outcome: dict[str, str] = {}
_kb: KnowledgeBase | None = None


def kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def _store(project_id: str) -> Store:
    with _lock:
        live = _running.get(project_id)
    if live is not None:
        return live
    store = Store(project_id)
    if store.version == 0:
        raise HTTPException(404, "No project with that id")
    return store


def _run(store: Store, images: list[bytes] | None) -> None:
    pid = store.project_id

    def on_event(kind: str, data: dict) -> None:
        with _lock:
            _events.setdefault(pid, []).append({"t": time.time(), "kind": kind, **data})

    try:
        llm = TrackedLLM(get_llm(on_retry=lambda info: on_event("retry", info)), on_event)
        _outcome[pid] = Orchestrator(store, llm, on_event=on_event, kb=kb()).run(images=images)
    except Exception as e:  # noqa: BLE001 - surface any failure in the dashboard
        _outcome[pid] = "error"
        on_event("error", {"message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
    finally:
        with _lock:
            _running.pop(pid, None)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "worker_model": config.WORKER_MODEL, "manager_model": config.MANAGER_MODEL,
            "vector_db": config.VECTOR_DB, "embeddings": config.EMBEDDINGS}


@app.get("/api/projects")
def projects():
    items = list_projects()
    with _lock:
        running = set(_running)
    for p in items:
        p["running"] = p["id"] in running
    return items


@app.post("/api/projects")
async def create_project(idea: str = Form(...), image: UploadFile | None = File(None)):
    idea = idea.strip()
    if len(idea) < 10:
        raise HTTPException(400, "Describe the product in at least a few words.")
    images = None
    if image is not None and image.filename:
        data = await image.read()
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(400, "Image is larger than 8 MB.")
        images = [data]
    store = Store.create(name=idea[:60], idea=idea)
    with _lock:
        _running[store.project_id] = store
        _events[store.project_id] = []
    threading.Thread(target=_run, args=(store, images), daemon=True).start()
    return {"id": store.project_id}


@app.post("/api/projects/{project_id}/resume")
def resume(project_id: str):
    """Continue a run that stopped (e.g. a Gemini API error) from the first unapproved phase."""
    with _lock:
        if project_id in _running:
            raise HTTPException(409, "This project is already running.")
    store = _store(project_id)
    if store.state["project"]["status"] == "completed":
        raise HTTPException(400, "This project is already complete.")
    with _lock:
        _running[project_id] = store
        _events.setdefault(project_id, []).append({"t": time.time(), "kind": "resumed"})
    threading.Thread(target=_run, args=(store, None), daemon=True).start()
    return {"id": project_id}


@app.get("/api/projects/{project_id}")
def project(project_id: str):
    store = _store(project_id)
    state = store.state
    log = ProcessLog(project_id, engine=store.engine).entries()
    for e in log:
        e["created_at"] = e["created_at"].isoformat() if e.get("created_at") else None
    history = [{"seq": a["seq"], "type": a["type"], "agent": a["agent"],
                "at": a["created_at"].isoformat() if a["created_at"] else None} for a in store.history()]
    with _lock:
        events = list(_events.get(project_id, []))
        running = project_id in _running
    return {"state": state, "log": log, "history": history[-300:], "events": events[-300:],
            "running": running, "outcome": _outcome.get(project_id), "phases": PHASE_ORDER}


@app.get("/api/projects/{project_id}/download")
def download(project_id: str):
    state = _store(project_id).state
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, src in {**state["code"]["files"], **state["tests"]["files"]}.items():
            z.writestr(path, src)
        reqs = "\n".join(f"- **{r['id']}** {r['title']}: {r['description']}"
                         for r in state["requirements"].values())
        z.writestr("README.md", f"# {state['project']['name']}\n\nGenerated by PipelineIQ.\n\n"
                                f"## Requirements\n{reqs}\n\n## Run the tests\n\n    pip install pytest\n    python -m pytest\n")
        z.writestr("conftest.py", "")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="pipelineiq-{project_id}.zip"'})


@app.get("/api/knowledge")
def knowledge():
    try:
        return {"collections": kb().stats()}
    except Exception as e:  # noqa: BLE001
        return {"collections": {}, "error": str(e)}
