import json

import pytest

from pipelineiq import test_runner
from pipelineiq.agents import AGENTS, PHASE_ORDER
from pipelineiq.db import get_engine
from pipelineiq.llm import FakeLLM
from pipelineiq.store import Store

CODE = '''class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
'''
TESTS = '''from app.calc import Calculator

def test_req_1_add():
    assert Calculator().add(2, 3) == 5
'''
BROKEN_CODE = CODE.replace("a + b", "a - b")

REPLIES = {
    "Requirements Agent": {"items": [{"id": "REQ-1", "title": "Add numbers", "description": "Add two ints",
                                      "acceptance_criteria": ["add(2, 3) returns 5"]}]},
    "Design Agent": {"design": {"summary": "A calculator", "modules": [{"path": "app/calc.py", "purpose": "math",
                                "requirement_ids": ["REQ-1"]}],
                                "interfaces": [{"name": "Calculator.add", "signature": "def add(self, a, b) -> int",
                                                "behaviour": "returns a + b", "requirement_ids": ["REQ-1"]}],
                                "data_model": []}},
    "Development Agent": {"files": {"app/__init__.py": "", "app/calc.py": CODE}},
    "Testing Agent": {"files": {"tests/test_app.py": TESTS}},
}


def fake_responder(prompt, system):
    for role, reply in REPLIES.items():
        if role in system:
            return "```json\n" + json.dumps(reply) + "\n```"
    raise AssertionError("unexpected agent")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(test_runner, "WORKSPACE", tmp_path / "ws")
    return Store.create("Calc", "A calculator", engine=get_engine(f"sqlite:///{tmp_path / 'a.db'}"))


def run_all(store, llm):
    for phase in PHASE_ORDER:
        tid = f"{phase.upper()}-1"
        store.dispatch("TASK_CREATED", "orchestrator", {"task_id": tid, "phase": phase, "agent": phase})
        AGENTS[phase](store, llm).run(tid)


def test_full_pipeline_with_fake_gemma(store):
    run_all(store, FakeLLM(fake_responder))
    s = store.state
    assert list(s["requirements"]) == ["REQ-1"]
    assert "app/calc.py" in s["code"]["files"]
    assert all(t["status"] == "submitted" for t in s["tasks"].values())
    results = test_runner.run_tests(store)
    assert results["passed"] == 1 and results["failed"] == 0
    assert results["cases"][0]["name"] == "test_req_1_add"


def test_failing_code_is_detected(store):
    REPLIES_BROKEN = dict(REPLIES, **{"Development Agent": {"files": {"app/calc.py": BROKEN_CODE}}})
    llm = FakeLLM(lambda p, s: json.dumps(next(v for k, v in REPLIES_BROKEN.items() if k in s)))
    run_all(store, llm)
    results = test_runner.run_tests(store)
    assert results["failed"] == 1


def test_invalid_json_gets_one_retry(store):
    calls = {"n": 0}

    def flaky(prompt, system):
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else json.dumps(REPLIES["Requirements Agent"])

    store.dispatch("TASK_CREATED", "orchestrator", {"task_id": "R1", "phase": "requirements", "agent": "requirements"})
    AGENTS["requirements"](store, FakeLLM(flaky)).run("R1")
    assert calls["n"] == 2 and "REQ-1" in store.state["requirements"]


def test_rework_feedback_reaches_the_prompt(store):
    llm = FakeLLM(fake_responder)
    store.dispatch("TASK_CREATED", "orchestrator", {"task_id": "R1", "phase": "requirements", "agent": "requirements"})
    AGENTS["requirements"](store, llm).run("R1")
    store.dispatch("TASK_REWORK_REQUESTED", "manager", {"task_id": "R1", "feedback": "Add a REQ for subtraction"})
    AGENTS["requirements"](store, llm).run("R1")
    assert "Add a REQ for subtraction" in llm.calls[-1]["prompt"]


def test_path_traversal_is_rejected(store):
    bad = {"files": {"../evil.py": "x"}}
    store.dispatch("TASK_CREATED", "orchestrator", {"task_id": "D1", "phase": "development", "agent": "development"})
    with pytest.raises(RuntimeError):
        AGENTS["development"](store, FakeLLM(lambda p, s: json.dumps(bad))).run("D1")


def test_workspace_inside_project_does_not_run_our_own_tests(tmp_path, monkeypatch):
    # Simulates the real layout: workspace/ lives inside the PipelineIQ folder, which has its own pytest.ini.
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_outer.py").write_text("def test_outer():\n    assert False\n")
    monkeypatch.setattr(test_runner, "WORKSPACE", tmp_path / "workspace")
    store = Store.create("Calc", "x", engine=get_engine(f"sqlite:///{tmp_path / 'b.db'}"))
    run_all(store, FakeLLM(fake_responder))
    results = test_runner.run_tests(store)
    assert [c["name"] for c in results["cases"]] == ["test_req_1_add"]
