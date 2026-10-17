"""Base class shared by all phase agents.

An agent:
  1. reads only the slices of shared state it needs (selector),
  2. asks Gemma 4 for a structured JSON result, guided by its system prompt,
  3. dispatches actions to the store with the result,
  4. submits its task for review by the Manager.
If the Manager asked for rework, the feedback is added to the prompt.
"""
from __future__ import annotations

import json

from ..llm import extract_json
from ..store import Store


class BaseAgent:
    name: str = "base"            # must match the agent name used in store.OWNERS
    phase: str = "base"
    reads: tuple[str, ...] = ()   # state slices this agent may read
    system_prompt: str = ""
    thinking: bool = False

    def __init__(self, store: Store, llm, model: str | None = None, kb=None):
        self.store = store
        self.llm = llm
        self.model = model
        self.kb = kb                    # KnowledgeBase; the agent searches its own collection
        self.retrieved: list[dict] = []

    def retrieval_query(self, context: dict) -> str:
        """What to search this agent's knowledge base for. Subclasses may refine it."""
        reqs = " ".join(r.get("title", "") for r in context.get("requirements", {}).values())
        return f"{self.phase}: {context['project']['idea']} {reqs}".strip()

    def knowledge_section(self, query: str) -> str:
        if self.kb is None:
            return ""
        from .. import config
        self.retrieved = self.kb.search(self.name, query, k=config.RAG_TOP_K)
        if not self.retrieved:
            return ""
        parts = [f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(self.retrieved, 1)]
        return ("\n\n## Retrieved knowledge (follow these team guidelines and avoid past mistakes)\n"
                + "\n\n".join(parts))

    # -- hooks for subclasses ---------------------------------------------
    def build_prompt(self, context: dict) -> str:
        raise NotImplementedError

    def apply(self, result: dict) -> None:
        """Dispatch actions to the store using the model's JSON result."""
        raise NotImplementedError

    def validate(self, result: dict) -> None:
        """Raise ValueError if the JSON does not have the expected shape."""

    # -- main entry point ---------------------------------------------------
    def run(self, task_id: str, images: list[bytes] | None = None) -> dict:
        context = self.store.select(*self.reads) if self.reads else {}
        context["project"] = self.store.select("project")["project"]
        task = self.store.state["tasks"].get(task_id, {})

        prompt = self.build_prompt(context)
        query = self.retrieval_query(context)
        if task.get("feedback"):
            query += " " + task["feedback"]
        prompt += self.knowledge_section(query)
        if task.get("status") == "rework" and task.get("feedback"):
            prompt += ("\n\n## Reviewer feedback on your previous attempt\n"
                       f"{task['feedback']}\n"
                       "Fix every point above. Return the complete corrected result, not a diff.")

        last_error = None
        for _ in range(2):  # one retry if the model returns malformed JSON
            # API errors (500, quota, ...) are retried inside the LLM client and propagate from here;
            # this loop only re-asks when the reply itself is malformed.
            text = self.llm.generate(prompt + "\n\nRespond with valid JSON only.", system=self.system_prompt,
                                     model=self.model, images=images, thinking=self.thinking)
            try:
                result = extract_json(text)
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object")
                self.validate(result)
                break
            except Exception as e:  # noqa: BLE001 - feed the error back to the model once
                last_error = e
                prompt += f"\n\nYour previous reply was invalid ({e}). Return only valid JSON in the required format."
        else:
            raise RuntimeError(f"{self.name} agent failed to produce valid output: {last_error}")

        self.apply(result)
        self.store.dispatch("TASK_SUBMITTED", self.name, {"task_id": task_id})
        return result


def as_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
