"""Orchestrator: runs the phases in order, sends each result to the Manager,
and follows the Manager's decision (next phase, rework, or stop for a human)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .agents import AGENTS, PHASE_ORDER
from .manager import Manager, Review
from .process_log import ProcessLog
from .rag import KnowledgeBase
from .store import Store

MAX_STEPS = 20  # hard safety cap on agent runs per project
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


class Orchestrator:
    def __init__(self, store: Store, llm, on_event: Callable[[str, dict], None] | None = None,
                 kb: KnowledgeBase | None = None):
        self.store = store
        self.llm = llm
        self.kb = kb or KnowledgeBase()
        self.log = ProcessLog(store.project_id, engine=store.engine)
        self.manager = Manager(store, self.log, llm, kb=self.kb)
        self.on_event = on_event or (lambda kind, data: None)

    def _task_id(self, phase: str) -> str:
        return f"{phase.upper()}-1"

    def _ensure_task(self, phase: str) -> dict:
        tid = self._task_id(phase)
        if tid not in self.store.state["tasks"]:
            self.store.dispatch("TASK_CREATED", "orchestrator", {"task_id": tid, "phase": phase, "agent": phase})
        return self.store.state["tasks"][tid]

    def _reopen_after(self, target: str, current: str) -> None:
        """Work earlier in the pipeline changed, so later finished tasks must be redone.
        Exception: a code fix requested by the testing phase keeps the existing tests."""
        start, end = PHASE_ORDER.index(target) + 1, PHASE_ORDER.index(current)
        for phase in PHASE_ORDER[start:end + 1]:
            if target == "development" and phase == "testing" and current == "testing":
                continue  # keep the tests; they will simply be re-run
            tid = self._task_id(phase)
            if tid in self.store.state["tasks"] and phase != target:
                self.store.dispatch("TASK_REOPENED", "orchestrator",
                                    {"task_id": tid, "reason": f"{target} was revised"})

    def run(self, images: list[bytes] | None = None) -> str:
        """Run the pipeline to completion. Returns 'completed' or 'escalated'."""
        added = self.kb.ingest_dir(KNOWLEDGE_DIR)  # only new/changed chunks are embedded
        self.on_event("knowledge", {"added": added, "stats": self.kb.stats()})
        # Resume support: start at the first phase that is not yet approved.
        tasks = self.store.state["tasks"]
        i = next((n for n, p in enumerate(PHASE_ORDER)
                  if tasks.get(self._task_id(p), {}).get("status") != "approved"), len(PHASE_ORDER))
        steps = 0
        while i < len(PHASE_ORDER):
            phase = PHASE_ORDER[i]
            steps += 1
            if steps > MAX_STEPS:
                tid = self._task_id(phase)
                self.store.dispatch("TASK_ESCALATED", "manager",
                                    {"task_id": tid, "feedback": "Step limit reached; human review needed."})
                self.on_event("escalated", {"phase": phase, "feedback": "Step limit reached"})
                return "escalated"

            self.store.dispatch("PHASE_CHANGED", "orchestrator", {"phase": phase})
            task = self._ensure_task(phase)

            # A task still 'submitted' means its output is unchanged and only needs re-review
            # (e.g. existing tests re-run after a code fix).
            if task["status"] != "submitted":
                self.on_event("agent_start", {"phase": phase, "attempt": task["attempt"]})
                imgs = images if phase == "requirements" else None
                agent = AGENTS[phase](self.store, self.llm, kb=self.kb)
                agent.run(task["task_id"], images=imgs)
                self.on_event("agent_done", {"phase": phase,
                                             "retrieved": [{"source": h["source"], "score": h["score"]}
                                                           for h in agent.retrieved]})
            else:
                self.on_event("recheck", {"phase": phase})

            review: Review = self.manager.review(task["task_id"])
            self.on_event("review", review.__dict__)

            if review.decision == "APPROVE":
                i += 1
            elif review.decision == "REWORK":
                target = review.route_to or phase
                self._reopen_after(target, phase)
                i = PHASE_ORDER.index(target)
            else:
                return "escalated"

        self.store.dispatch("PROJECT_COMPLETED", "orchestrator", {})
        return "completed"
