"""Manager + Orchestrator tests with a scripted fake Gemma (no API key needed)."""
import json

import pytest

from pipelineiq import config, test_runner
from pipelineiq.db import get_engine
from pipelineiq.llm import FakeLLM
from pipelineiq.orchestrator import Orchestrator
from pipelineiq.store import Store

GOOD_CODE = "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
BAD_CODE = GOOD_CODE.replace("a + b", "a - b")
TESTS = ("from app.calc import Calculator\n\n"
         "def test_req_1_add():\n    assert Calculator().add(2, 3) == 5\n\n"
         "def test_req_2_add_zero():\n    assert Calculator().add(2, 0) == 2\n\n"
         "def test_req_3_add_negative():\n    assert Calculator().add(-1, -1) == -2\n")
REQS = {"items": [{"id": f"REQ-{i}", "title": t, "description": t, "acceptance_criteria": ["works"]}
                  for i, t in enumerate(["add", "add zero", "add negatives"], 1)]}
DESIGN = {"design": {"summary": "calc",
                     "modules": [{"path": "app/calc.py", "purpose": "math", "requirement_ids": ["REQ-1", "REQ-2", "REQ-3"]}],
                     "interfaces": [{"name": "Calculator.add", "signature": "add(a,b)", "behaviour": "a+b",
                                     "requirement_ids": ["REQ-1"]}], "data_model": []}}
APPROVE = {"checks": [{"name": "ok", "result": "PASS"}], "score": 90, "decision": "APPROVE", "feedback": ""}


class Script:
    """Fake Gemma whose development replies can change between attempts."""

    def __init__(self, dev_codes, manager_override=None):
        self.dev_codes = list(dev_codes)
        self.manager_override = manager_override

    def __call__(self, prompt, system):
        if "ROOT_CAUSE" in prompt:
            return json.dumps({"route_to": "development", "feedback": "add() must return a + b"})
        if "Manager Agent" in system:
            return json.dumps(self.manager_override(prompt) if self.manager_override else APPROVE)
        if "Requirements Agent" in system:
            return json.dumps(REQS)
        if "Design Agent" in system:
            return json.dumps(DESIGN)
        if "Development Agent" in system:
            code = self.dev_codes.pop(0) if len(self.dev_codes) > 1 else self.dev_codes[0]
            return json.dumps({"files": {"app/calc.py": code}})
        if "Testing Agent" in system:
            return json.dumps({"files": {"tests/test_app.py": TESTS}})
        raise AssertionError(system)


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(test_runner, "WORKSPACE", tmp_path / "ws")

    def _make(script):
        store = Store.create("Calc", "a calculator", engine=get_engine(f"sqlite:///{tmp_path / 'm.db'}"))
        events = []
        orch = Orchestrator(store, FakeLLM(script), on_event=lambda k, d: events.append((k, d)))
        return store, orch, events
    return _make


def test_happy_path_completes(make):
    store, orch, _ = make(Script([GOOD_CODE]))
    assert orch.run() == "completed"
    assert store.state["project"]["status"] == "completed"
    assert [e["decision"] for e in orch.log.entries()] == ["APPROVE"] * 4


def test_failing_tests_route_back_to_development_then_pass(make):
    store, orch, events = make(Script([BAD_CODE, GOOD_CODE]))
    assert orch.run() == "completed"
    log = [(e["phase"], e["decision"]) for e in orch.log.entries()]
    assert log == [("requirements", "APPROVE"), ("design", "APPROVE"), ("development", "APPROVE"),
                   ("testing", "REWORK"), ("development", "APPROVE"), ("testing", "APPROVE")]
    assert store.state["tasks"]["DEVELOPMENT-1"]["attempt"] == 2
    # the tests were kept and re-run, not regenerated
    starts = [d["phase"] for k, d in events if k == "agent_start"]
    assert starts.count("testing") == 1
    assert store.state["tests"]["results"]["failed"] == 0


def test_escalates_after_retry_limit(make, monkeypatch):
    monkeypatch.setattr(config, "MAX_REWORK_ATTEMPTS", 2)
    store, orch, _ = make(Script([BAD_CODE]))  # never fixed
    assert orch.run() == "escalated"
    assert orch.log.entries()[-1]["decision"] == "ESCALATE"
    assert store.state["project"]["status"] == "needs_human"


def test_llm_rework_feedback_reaches_agent(make):
    calls = {"n": 0}

    def manager(prompt):
        if "## Work to review (requirements)" in prompt:
            calls["n"] += 1
            if calls["n"] == 1:
                return {"checks": [{"name": "Specific", "result": "FAIL", "reason": "vague"}],
                        "score": 40, "decision": "REWORK", "feedback": "1. Make REQ-1 measurable"}
        return APPROVE

    store, orch, _ = make(Script([GOOD_CODE], manager_override=manager))
    assert orch.run() == "completed"
    first = orch.log.entries()[0]
    assert first["decision"] == "REWORK" and first["score"] == 40
    assert store.state["tasks"]["REQUIREMENTS-1"]["attempt"] == 2


def test_design_coverage_check_is_deterministic(make):
    bad_design = json.loads(json.dumps(DESIGN))
    bad_design["design"]["modules"][0]["requirement_ids"] = ["REQ-1"]
    script = Script([GOOD_CODE])
    replies = {"n": 0}
    original = script.__call__

    def patched(prompt, system):
        if "Design Agent" in system:
            replies["n"] += 1
            return json.dumps(bad_design if replies["n"] == 1 else DESIGN)
        return original(prompt, system)

    store, orch, _ = make(patched)
    assert orch.run() == "completed"
    design_reviews = [e for e in orch.log.entries() if e["phase"] == "design"]
    assert design_reviews[0]["decision"] == "REWORK"
    assert "REQ-2" in design_reviews[0]["feedback"]


def test_rework_feedback_becomes_a_lesson(make):
    store, orch, _ = make(Script([BAD_CODE, GOOD_CODE]))
    orch.run()
    hits = orch.kb.search("development", "add must return a + b")
    assert any(h["source"] == "lessons-learned" for h in hits)


def test_resume_skips_approved_phases(make):
    class Boom(Exception):
        pass

    script = Script([GOOD_CODE])
    state = {"fail": True}

    def flaky(prompt, system):
        if "Development Agent" in system and state["fail"]:
            raise Boom("500 INTERNAL")
        return script(prompt, system)

    store, orch, events = make(flaky)
    with pytest.raises(Boom):
        orch.run()
    state["fail"] = False
    events.clear()
    assert orch.run() == "completed"
    started = [d["phase"] for k, d in events if k == "agent_start"]
    assert started[0] == "development"          # requirements and design were not redone
