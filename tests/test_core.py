import pytest

from pipelineiq.db import get_engine
from pipelineiq.llm import FakeLLM, extract_json
from pipelineiq.process_log import ProcessLog
from pipelineiq.store import PermissionDenied, Store, VersionConflict


@pytest.fixture
def engine(tmp_path):
    return get_engine(f"sqlite:///{tmp_path / 'test.db'}")


def test_create_and_dispatch(engine):
    store = Store.create("Todo API", "Build a to-do API with OTP login", engine=engine)
    assert store.state["project"]["phase"] == "requirements"
    store.dispatch("REQUIREMENTS_SET", "requirements",
                   {"items": [{"id": "REQ-1", "title": "Login with OTP"}]})
    assert "REQ-1" in store.state["requirements"]
    assert store.version == 2


def test_ownership_is_enforced(engine):
    store = Store.create("x", "y", engine=engine)
    with pytest.raises(PermissionDenied):
        store.dispatch("REQUIREMENTS_SET", "development", {"items": []})


def test_optimistic_locking(engine):
    store = Store.create("x", "y", engine=engine)
    seen = store.version
    store.dispatch("PHASE_CHANGED", "orchestrator", {"phase": "design"})
    with pytest.raises(VersionConflict):
        store.dispatch("PHASE_CHANGED", "orchestrator", {"phase": "development"},
                       expected_version=seen)


def test_state_is_rebuilt_from_action_log(engine):
    store = Store.create("x", "y", engine=engine)
    store.dispatch("CODE_FILES_WRITTEN", "development", {"files": {"app.py": "print(1)"}})
    reloaded = Store(store.project_id, engine=engine)
    assert reloaded.state == store.state
    assert [a["type"] for a in reloaded.history()] == ["PROJECT_CREATED", "CODE_FILES_WRITTEN"]


def test_subscribers_are_notified(engine):
    store = Store.create("x", "y", engine=engine)
    seen = []
    unsubscribe = store.subscribe(lambda a, s: seen.append(a["type"]), types=["DESIGN_SET"])
    store.dispatch("PHASE_CHANGED", "orchestrator", {"phase": "design"})
    store.dispatch("DESIGN_SET", "design", {"design": {"summary": "REST API"}})
    unsubscribe()
    store.dispatch("DESIGN_SET", "design", {"design": {"summary": "v2"}})
    assert seen == ["DESIGN_SET"]


def test_task_lifecycle_and_rework(engine):
    store = Store.create("x", "y", engine=engine)
    store.dispatch("TASK_CREATED", "orchestrator", {"task_id": "T1", "phase": "design", "agent": "design"})
    with pytest.raises(PermissionDenied):
        store.dispatch("TASK_SUBMITTED", "testing", {"task_id": "T1"})
    store.dispatch("TASK_SUBMITTED", "design", {"task_id": "T1"})
    store.dispatch("TASK_REWORK_REQUESTED", "manager", {"task_id": "T1", "feedback": "Add APIs"})
    task = store.state["tasks"]["T1"]
    assert task["status"] == "rework" and task["attempt"] == 2


def test_process_log(engine):
    log = ProcessLog("p1", engine=engine)
    log.record(task_id="T1", phase="design", agent="design", attempt=1,
               checks=[{"name": "APIs defined", "result": "FAIL"}], decision="REWORK",
               feedback="Add APIs", score=60)
    log.record(task_id="T1", phase="design", agent="design", attempt=2,
               checks=[{"name": "APIs defined", "result": "PASS"}], decision="APPROVE", score=90)
    assert [e["status"] for e in log.entries("T1")] == ["FAIL", "PASS"]
    assert log.summary() == {"design": {"APPROVE": 1, "REWORK": 1, "ESCALATE": 0}}


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: [1, 2] thanks') == [1, 2]
    assert FakeLLM(lambda p, s: '{"x": 2}').generate_json("hi") == {"x": 2}
