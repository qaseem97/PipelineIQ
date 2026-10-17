"""Redux-style shared state store for PipelineIQ.

Agents never edit the state directly. They dispatch actions; a reducer applies
each action to produce the next state. Every action is persisted, so the full
state can be rebuilt (and audited) by replaying the action log.

    store = Store.create("Todo API", "Build a to-do API with OTP login")
    store.subscribe(lambda action, state: print(action["type"]))
    store.dispatch("REQUIREMENTS_SET", "requirements", {"items": [...]})
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Callable, Iterable

from sqlalchemy import insert, select

from .db import actions_table, get_engine, now

Listener = Callable[[dict, dict], None]


class StoreError(Exception):
    """Base class for store errors."""


class PermissionDenied(StoreError):
    """An agent tried to dispatch an action it does not own."""


class VersionConflict(StoreError):
    """The state changed since the caller last read it (optimistic locking)."""


class UnknownAction(StoreError):
    """No reducer exists for this action type."""


# ---------------------------------------------------------------------------
# Ownership: which agents may dispatch which actions (single-writer rule).
# ---------------------------------------------------------------------------
OWNERS: dict[str, set[str]] = {
    "PROJECT_CREATED": {"orchestrator"},
    "PHASE_CHANGED": {"orchestrator"},
    "PROJECT_COMPLETED": {"orchestrator"},
    "TASK_CREATED": {"orchestrator"},
    "TASK_REOPENED": {"orchestrator"},
    "REQUIREMENTS_SET": {"requirements"},
    "DESIGN_SET": {"design"},
    "CODE_FILES_WRITTEN": {"development"},
    "TEST_FILES_WRITTEN": {"testing"},
    "TEST_RESULTS_RECORDED": {"testing", "test_runner"},
    "TASK_SUBMITTED": {"requirements", "design", "development", "testing"},
    "TASK_APPROVED": {"manager"},
    "TASK_REWORK_REQUESTED": {"manager"},
    "TASK_ESCALATED": {"manager"},
}


def initial_state(project_id: str) -> dict:
    return {
        "version": 0,
        "project": {"id": project_id, "name": "", "idea": "", "phase": "new", "status": "active"},
        "requirements": {},                       # REQ-1 -> {...}
        "design": {},                             # architecture, components, apis, data model
        "code": {"files": {}},                    # path -> source
        "tests": {"files": {}, "results": None},  # path -> source, last run results
        "tasks": {},                              # task_id -> {phase, agent, status, attempt, feedback}
    }


# ---------------------------------------------------------------------------
# Reducers: pure functions (state, payload, agent) -> new state
# ---------------------------------------------------------------------------
def _project_created(s, p, a):
    s["project"].update(name=p["name"], idea=p["idea"], phase="requirements")
    return s


def _phase_changed(s, p, a):
    s["project"]["phase"] = p["phase"]
    return s


def _project_completed(s, p, a):
    s["project"].update(phase="done", status="completed")
    return s


def _task_created(s, p, a):
    s["tasks"][p["task_id"]] = {"task_id": p["task_id"], "phase": p["phase"], "agent": p["agent"],
                                "status": "in_progress", "attempt": 1, "feedback": None}
    return s


def _task_reopened(s, p, a):
    """Upstream work changed, so this task must be redone (not counted as a failed attempt)."""
    _task(s, p).update(status="in_progress", feedback=p.get("reason"))
    return s


def _requirements_set(s, p, a):
    s["requirements"] = {r["id"]: r for r in p["items"]}
    return s


def _design_set(s, p, a):
    s["design"] = p["design"]
    return s


def _code_files_written(s, p, a):
    s["code"]["files"].update(p["files"])
    return s


def _test_files_written(s, p, a):
    s["tests"]["files"].update(p["files"])
    return s


def _test_results_recorded(s, p, a):
    s["tests"]["results"] = p
    return s


def _task(s, p):
    task = s["tasks"].get(p["task_id"])
    if task is None:
        raise StoreError(f"Unknown task {p['task_id']}")
    return task


def _task_submitted(s, p, a):
    task = _task(s, p)
    if task["agent"] != a:
        raise PermissionDenied(f"{a} cannot submit task owned by {task['agent']}")
    task["status"] = "submitted"
    return s


def _task_approved(s, p, a):
    _task(s, p).update(status="approved", feedback=p.get("feedback"))
    return s


def _task_rework(s, p, a):
    task = _task(s, p)
    task.update(status="rework", feedback=p.get("feedback"), attempt=task["attempt"] + 1)
    return s


def _task_escalated(s, p, a):
    _task(s, p).update(status="escalated", feedback=p.get("feedback"))
    s["project"]["status"] = "needs_human"
    return s


REDUCERS: dict[str, Callable[[dict, dict, str], dict]] = {
    "PROJECT_CREATED": _project_created,
    "PHASE_CHANGED": _phase_changed,
    "PROJECT_COMPLETED": _project_completed,
    "TASK_CREATED": _task_created,
    "TASK_REOPENED": _task_reopened,
    "REQUIREMENTS_SET": _requirements_set,
    "DESIGN_SET": _design_set,
    "CODE_FILES_WRITTEN": _code_files_written,
    "TEST_FILES_WRITTEN": _test_files_written,
    "TEST_RESULTS_RECORDED": _test_results_recorded,
    "TASK_SUBMITTED": _task_submitted,
    "TASK_APPROVED": _task_approved,
    "TASK_REWORK_REQUESTED": _task_rework,
    "TASK_ESCALATED": _task_escalated,
}


def reduce(state: dict, action: dict) -> dict:
    """Apply one action to a copy of the state and return the new state."""
    reducer = REDUCERS.get(action["type"])
    if reducer is None:
        raise UnknownAction(action["type"])
    new_state = reducer(copy.deepcopy(state), action["payload"], action["agent"])
    new_state["version"] = state["version"] + 1
    return new_state


def list_projects(engine=None, limit: int = 50) -> list[dict]:
    """Newest first: id, name, idea and creation time of each project."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(actions_table.c.project_id, actions_table.c.payload, actions_table.c.created_at)
            .where(actions_table.c.type == "PROJECT_CREATED")
            .order_by(actions_table.c.id.desc()).limit(limit)
        ).mappings().all()
    return [{"id": r["project_id"], "name": r["payload"]["name"], "idea": r["payload"]["idea"],
             "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class Store:
    def __init__(self, project_id: str, engine=None):
        self.project_id = project_id
        self.engine = engine or get_engine()
        self._state = initial_state(project_id)
        self._listeners: list[tuple[Listener, set[str] | None]] = []
        self._replay()

    # -- construction ------------------------------------------------------
    @classmethod
    def create(cls, name: str, idea: str, engine=None) -> "Store":
        store = cls(uuid.uuid4().hex[:12], engine=engine)
        store.dispatch("PROJECT_CREATED", "orchestrator", {"name": name, "idea": idea})
        return store

    def _replay(self) -> None:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(actions_table)
                .where(actions_table.c.project_id == self.project_id)
                .order_by(actions_table.c.seq)
            ).mappings().all()
        for row in rows:
            self._state = reduce(self._state, {"type": row["type"], "agent": row["agent"],
                                               "payload": row["payload"]})

    # -- reading -----------------------------------------------------------
    @property
    def state(self) -> dict:
        """A copy of the full state (callers cannot mutate the store by accident)."""
        return copy.deepcopy(self._state)

    @property
    def version(self) -> int:
        return self._state["version"]

    def select(self, *keys: str) -> dict:
        """Selector: return only the slices an agent needs, e.g. select('requirements', 'design')."""
        return {k: copy.deepcopy(self._state[k]) for k in keys}

    def history(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(actions_table)
                .where(actions_table.c.project_id == self.project_id)
                .order_by(actions_table.c.seq)
            ).mappings().all()
        return [dict(r) for r in rows]

    # -- writing -----------------------------------------------------------
    def dispatch(self, type: str, agent: str, payload: dict[str, Any] | None = None,
                 expected_version: int | None = None) -> dict:
        """Validate, persist and apply an action, then notify subscribers.

        expected_version enables optimistic locking: pass the version you read,
        and the dispatch is rejected if someone else changed the state since.
        """
        payload = payload or {}
        if type not in REDUCERS:
            raise UnknownAction(type)
        if agent not in OWNERS.get(type, set()):
            raise PermissionDenied(f"Agent '{agent}' is not allowed to dispatch {type}")
        if expected_version is not None and expected_version != self.version:
            raise VersionConflict(f"Expected version {expected_version}, store is at {self.version}")

        action = {"type": type, "agent": agent, "payload": payload}
        new_state = reduce(self._state, action)   # raises before anything is saved if invalid

        with self.engine.begin() as conn:
            conn.execute(insert(actions_table).values(
                project_id=self.project_id, seq=new_state["version"], type=type,
                agent=agent, payload=payload, created_at=now()))

        self._state = new_state
        for listener, types in list(self._listeners):
            if types is None or type in types:
                listener(action, self.state)
        return action

    def subscribe(self, listener: Listener, types: Iterable[str] | None = None) -> Callable[[], None]:
        """Call listener(action, state) after matching actions. Returns an unsubscribe function."""
        entry = (listener, set(types) if types else None)
        self._listeners.append(entry)
        return lambda: self._listeners.remove(entry)
