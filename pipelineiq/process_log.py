"""Append-only process log: one entry per Manager review (pass/fail, checks, decision)."""
from __future__ import annotations

from sqlalchemy import insert, select

from .db import get_engine, now, process_log_table

DECISIONS = {"APPROVE", "REWORK", "ESCALATE"}


class ProcessLog:
    def __init__(self, project_id: str, engine=None):
        self.project_id = project_id
        self.engine = engine or get_engine()

    def record(self, *, task_id: str, phase: str, agent: str, attempt: int,
               checks: list[dict], decision: str, feedback: str | None = None,
               score: int | None = None) -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        status = "PASS" if decision == "APPROVE" else "FAIL"
        entry = dict(project_id=self.project_id, task_id=task_id, phase=phase, agent=agent,
                     attempt=attempt, status=status, score=score, checks=checks,
                     decision=decision, feedback=feedback, created_at=now())
        with self.engine.begin() as conn:
            result = conn.execute(insert(process_log_table).values(**entry))
            entry["id"] = result.inserted_primary_key[0]
        return entry

    def entries(self, task_id: str | None = None) -> list[dict]:
        q = select(process_log_table).where(process_log_table.c.project_id == self.project_id)
        if task_id:
            q = q.where(process_log_table.c.task_id == task_id)
        with self.engine.connect() as conn:
            rows = conn.execute(q.order_by(process_log_table.c.id)).mappings().all()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        """Counts per phase and decision, handy for the dashboard."""
        out: dict[str, dict[str, int]] = {}
        for e in self.entries():
            phase = out.setdefault(e["phase"], {"APPROVE": 0, "REWORK": 0, "ESCALATE": 0})
            phase[e["decision"]] += 1
        return out
