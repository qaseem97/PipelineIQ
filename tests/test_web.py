import time

import pytest
from fastapi.testclient import TestClient

from pipelineiq import db, test_runner, web
from pipelineiq.llm import FakeLLM
from test_manager import BAD_CODE, GOOD_CODE, Script


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_engine", db.get_engine(f"sqlite:///{tmp_path / 'w.db'}"))
    monkeypatch.setattr(test_runner, "WORKSPACE", tmp_path / "ws")
    monkeypatch.setattr(web, "_kb", None)
    monkeypatch.setattr(web, "get_llm", lambda **kw: FakeLLM(Script([BAD_CODE, GOOD_CODE])))
    return TestClient(web.app)


def wait_done(client, pid, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get(f"/api/projects/{pid}").json()
        if not d["running"]:
            return d
        time.sleep(0.2)
    raise AssertionError("pipeline did not finish")


def test_dashboard_runs_a_project_end_to_end(client):
    assert client.get("/").status_code == 200
    pid = client.post("/api/projects", data={"idea": "A calculator that adds numbers"}).json()["id"]
    d = wait_done(client, pid)
    assert d["outcome"] == "completed"
    assert [e["decision"] for e in d["log"]].count("REWORK") == 1
    assert {"knowledge", "agent_start", "review_start", "review"} <= {e["kind"] for e in d["events"]}
    started = {e["call"] for e in d["events"] if e["kind"] == "llm_start"}
    ended = {e["call"]: e for e in d["events"] if e["kind"] == "llm_end"}
    assert started and started == set(ended) and all(e["ok"] for e in ended.values())
    assert pid in [p["id"] for p in client.get("/api/projects").json()]
    z = client.get(f"/api/projects/{pid}/download")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"


def test_rejects_too_short_idea(client):
    assert client.post("/api/projects", data={"idea": "app"}).status_code == 400


def test_unknown_project_is_404(client):
    assert client.get("/api/projects/nope").status_code == 404


def test_resume_endpoint_rejects_completed_project(client):
    pid = client.post("/api/projects", data={"idea": "A calculator that adds numbers"}).json()["id"]
    wait_done(client, pid)
    assert client.post(f"/api/projects/{pid}/resume").status_code == 400
