"""Manager Agent: the quality gate after every task.

Review order:
  1. Deterministic checks (objective, run first). Any failure -> REWORK without asking the LLM.
  2. LLM review on Gemma 4 (larger model, thinking on) against a fixed rubric for the phase.
  3. Decision: APPROVE, REWORK (with specific feedback, routed to the phase at fault),
     or ESCALATE when the retry limit is reached.
Every review is written to the process log and dispatched to the shared store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .agents.base import as_json
from .process_log import ProcessLog
from .store import Store
from .test_runner import run_tests

RUBRICS = {
    "requirements": [
        "Every requirement is specific and unambiguous",
        "Every acceptance criterion can be verified by an automated test",
        "Together the requirements cover the core of the product idea",
        "Scope is small enough for a first version",
    ],
    "design": [
        "Every requirement is addressed by at least one interface",
        "Interfaces have clear signatures and described error behaviour",
        "Modules have a single, clear responsibility",
        "The design is implementable in pure standard-library Python",
    ],
    "development": [
        "The code implements every interface in the design with matching names and signatures",
        "Invalid input is handled as the design describes",
        "The code is readable and has no obvious bugs",
    ],
    "testing": [
        "Every acceptance criterion is checked by at least one test",
        "Tests check behaviour through public interfaces, not implementation details",
        "Edge cases and invalid input are tested where requirements mention them",
    ],
}

SYSTEM = (
    "You are the Manager Agent in PipelineIQ, an independent reviewer and quality gate. "
    "You did not write the work you review. Judge it strictly against the rubric and the "
    "requirements. Approve work that meets the rubric; do not block on minor style preferences. "
    "When you request rework, give specific, actionable instructions the author can follow."
)


@dataclass
class Review:
    task_id: str
    phase: str
    attempt: int
    decision: str                       # APPROVE | REWORK | ESCALATE
    checks: list[dict] = field(default_factory=list)
    score: int | None = None
    feedback: str = ""
    route_to: str | None = None         # phase that must redo work (defaults to the reviewed phase)


def _check(name: str, ok: bool, reason: str = "") -> dict:
    return {"name": name, "result": "PASS" if ok else "FAIL", "reason": "" if ok else reason}


def _req_number(req_id: str) -> str:
    m = re.search(r"(\d+)", req_id)
    return m.group(1) if m else req_id


class Manager:
    def __init__(self, store: Store, log: ProcessLog, llm, model: str | None = None, kb=None):
        self.store, self.log, self.llm, self.kb = store, log, llm, kb
        self.model = model or config.MANAGER_MODEL

    # -- deterministic checks ----------------------------------------------
    def deterministic_checks(self, phase: str, state: dict) -> list[dict]:
        reqs = state["requirements"]
        if phase == "requirements":
            ids = list(reqs)
            return [
                _check("At least 3 requirements", len(ids) >= 3, f"only {len(ids)} requirement(s)"),
                _check("Every requirement has acceptance criteria",
                       all(r.get("acceptance_criteria") for r in reqs.values()),
                       "some requirements have no acceptance criteria"),
            ]
        if phase == "design":
            covered = {rid for item in state["design"].get("modules", []) + state["design"].get("interfaces", [])
                       for rid in item.get("requirement_ids", [])}
            missing = [r for r in reqs if r not in covered]
            return [_check("Every requirement is covered by the design", not missing,
                           f"not covered: {', '.join(missing)}")]
        if phase == "development":
            files = state["code"]["files"]
            broken = []
            for path, src in files.items():
                try:
                    compile(src, path, "exec")
                except SyntaxError as e:
                    broken.append(f"{path} line {e.lineno}: {e.msg}")
            expected = [m["path"] for m in state["design"].get("modules", [])]
            absent = [p for p in expected if p not in files]
            return [
                _check("All files compile", not broken, "; ".join(broken)),
                _check("Every module in the design exists", not absent, f"missing: {', '.join(absent)}"),
            ]
        if phase == "testing":
            results = run_tests(self.store)
            names = [c["name"] for c in results["cases"]]
            untested = [r for r in reqs if not any(f"req_{_req_number(r)}_" in n or n.endswith(f"req_{_req_number(r)}")
                                                   for n in names)]
            return [
                _check("Tests ran", results["total"] > 0, "no tests were collected (import or syntax error?)"),
                _check("All tests pass", results["failed"] == 0 and results["total"] > 0,
                       f"{results['failed']} failing of {results['total']}"),
                _check("Every requirement has a test", not untested, f"no test for: {', '.join(untested)}"),
            ]
        return []

    # -- LLM review ------------------------------------------------------------
    def _work_for(self, phase: str, state: dict) -> str:
        return {
            "requirements": as_json(list(state["requirements"].values())),
            "design": as_json(state["design"]),
            "development": as_json(state["code"]["files"]),
            "testing": as_json(state["tests"]["files"]),
        }[phase]

    def llm_review(self, phase: str, state: dict) -> dict:
        context = f"Product idea: {state['project']['idea']}\n"
        if phase != "requirements":
            context += f"\nRequirements:\n{as_json(list(state['requirements'].values()))}\n"
        if phase in ("development", "testing"):
            context += f"\nDesign:\n{as_json(state['design'])}\n"
        if phase == "testing":
            context += f"\nCode under test:\n{as_json(state['code']['files'])}\n"
        rubric = "\n".join(f"- {r}" for r in RUBRICS[phase])
        if self.kb is not None:
            hits = self.kb.search("manager", f"reviewing {phase}: {state['project']['idea']}", k=3)
            if hits:
                rubric += "\n\n## Review guidelines (retrieved)\n" + "\n\n".join(h["text"] for h in hits)
        prompt = f"""{context}
## Work to review ({phase})
{self._work_for(phase, state)}

## Rubric
{rubric}

Return JSON:
{{
  "checks": [{{"name": "<rubric item>", "result": "PASS" or "FAIL", "reason": "why, if FAIL"}}],
  "score": <0-100>,
  "decision": "APPROVE" or "REWORK",
  "feedback": "numbered, specific fixes if REWORK; empty if APPROVE"
}}"""
        return self.llm.generate_json(prompt, system=SYSTEM, model=self.model, thinking=True)

    def find_root_cause(self, state: dict) -> dict:
        """Tests failed: decide whether the code or the tests are wrong."""
        prompt = f"""ROOT_CAUSE analysis. Generated tests failed.

Requirements:
{as_json(list(state['requirements'].values()))}

Code:
{as_json(state['code']['files'])}

Tests:
{as_json(state['tests']['files'])}

Test output (truncated):
{(state['tests']['results'] or {}).get('output', '')[-3000:]}

Decide which is wrong relative to the requirements: the code ("development") or the tests ("testing").
Return JSON: {{"route_to": "development" or "testing", "feedback": "specific fixes for that agent"}}"""
        result = self.llm.generate_json(prompt, system=SYSTEM, model=self.model, thinking=True)
        if result.get("route_to") not in ("development", "testing"):
            result["route_to"] = "development"
        return result

    # -- main entry point ------------------------------------------------------
    def review(self, task_id: str) -> Review:
        task = self.store.state["tasks"][task_id]
        phase, attempt = task["phase"], task["attempt"]
        checks = self.deterministic_checks(phase, self.store.state)
        state = self.store.state  # re-read: the testing checks record new results
        failed = [c for c in checks if c["result"] == "FAIL"]
        score, route_to = None, phase

        if failed:
            feedback = "\n".join(f"{i}. {c['name']}: {c['reason']}" for i, c in enumerate(failed, 1))
            if phase == "testing" and any(c["name"] == "All tests pass" for c in failed):
                rc = self.find_root_cause(state)
                route_to, feedback = rc["route_to"], rc.get("feedback") or feedback
            decision = "REWORK"
        else:
            verdict = self.llm_review(phase, state)
            checks += [dict(c, name=f"[LLM] {c.get('name', '')}") for c in verdict.get("checks", [])]
            score = verdict.get("score")
            decision = "APPROVE" if str(verdict.get("decision", "")).upper() == "APPROVE" else "REWORK"
            feedback = verdict.get("feedback", "") or ""

        target_task_id = self._task_for(route_to) or task_id
        target_attempt = self.store.state["tasks"][target_task_id]["attempt"]
        if decision == "REWORK" and target_attempt >= config.MAX_REWORK_ATTEMPTS:
            decision = "ESCALATE"
            feedback = f"Retry limit reached ({config.MAX_REWORK_ATTEMPTS}). Last feedback:\n{feedback}"

        self.log.record(task_id=task_id, phase=phase, agent=task["agent"], attempt=attempt,
                        checks=checks, decision=decision, feedback=feedback, score=score)
        if decision == "APPROVE":
            self.store.dispatch("TASK_APPROVED", "manager", {"task_id": task_id})
        elif decision == "REWORK":
            self.store.dispatch("TASK_REWORK_REQUESTED", "manager", {"task_id": target_task_id, "feedback": feedback})
            if self.kb is not None and feedback:
                # Lessons learned: future projects retrieve this feedback, so agents stop repeating mistakes.
                self.kb.add_lesson(route_to, self.store.state["project"]["idea"], feedback)
        else:
            self.store.dispatch("TASK_ESCALATED", "manager", {"task_id": task_id, "feedback": feedback})
        return Review(task_id, phase, attempt, decision, checks, score, feedback,
                      route_to if decision == "REWORK" else None)

    def _task_for(self, phase: str) -> str | None:
        for tid, t in self.store.state["tasks"].items():
            if t["phase"] == phase:
                return tid
        return None
