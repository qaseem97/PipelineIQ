"""The four phase agents built for the hackathon MVP."""
from __future__ import annotations

from .base import BaseAgent, as_json


class RequirementsAgent(BaseAgent):
    name = phase = "requirements"
    system_prompt = (
        "You are the Requirements Agent in PipelineIQ, an automated software development team. "
        "You turn a short product idea (and optional images such as whiteboard sketches) into clear, "
        "testable requirements. Every requirement must be specific, unambiguous and verifiable by an "
        "automated test. Keep scope small enough to build in one session."
    )

    def build_prompt(self, ctx):
        return f"""Product idea: {ctx['project']['idea']}

Write 3 to 6 functional requirements for a minimal but complete first version.

Return JSON in exactly this format:
{{
  "items": [
    {{
      "id": "REQ-1",
      "title": "short title",
      "description": "what the system must do",
      "acceptance_criteria": ["concrete, testable condition", "..."]
    }}
  ]
}}
IDs must be REQ-1, REQ-2, ... in order."""

    def validate(self, r):
        items = r.get("items")
        if not items or not isinstance(items, list):
            raise ValueError("'items' must be a non-empty list")
        for it in items:
            for key in ("id", "title", "description", "acceptance_criteria"):
                if key not in it:
                    raise ValueError(f"requirement missing '{key}'")

    def apply(self, r):
        self.store.dispatch("REQUIREMENTS_SET", self.name, {"items": r["items"]})


class DesignAgent(BaseAgent):
    name = phase = "design"
    reads = ("requirements",)
    system_prompt = (
        "You are the Design Agent in PipelineIQ. You design simple, clean software that satisfies "
        "the given requirements. Implementation will be pure Python 3.11 using only the standard "
        "library (no web frameworks, no databases, no network): model the system as Python modules, "
        "classes and functions. Every requirement must be covered by at least one interface."
    )

    def build_prompt(self, ctx):
        return f"""Requirements:
{as_json(list(ctx['requirements'].values()))}

Produce the software design.

Return JSON in exactly this format:
{{
  "design": {{
    "summary": "2-3 sentence overview",
    "modules": [
      {{"path": "app/<name>.py", "purpose": "...", "requirement_ids": ["REQ-1"]}}
    ],
    "interfaces": [
      {{"name": "ClassName.method or function", "signature": "def method(self, x: str) -> dict",
        "behaviour": "what it does, including errors raised", "requirement_ids": ["REQ-1"]}}
    ],
    "data_model": [{{"name": "...", "fields": {{"field": "type"}}}}]
  }}
}}"""

    def validate(self, r):
        d = r.get("design")
        if not isinstance(d, dict) or not d.get("modules") or not d.get("interfaces"):
            raise ValueError("'design' must include non-empty 'modules' and 'interfaces'")

    def apply(self, r):
        self.store.dispatch("DESIGN_SET", self.name, {"design": r["design"]})


class DevelopmentAgent(BaseAgent):
    name = phase = "development"
    reads = ("requirements", "design", "code", "tests")
    system_prompt = (
        "You are the Development Agent in PipelineIQ. You write correct, readable Python 3.11 code "
        "that implements the design exactly: same module paths, class names and signatures. "
        "Use only the standard library. Include an empty app/__init__.py. Raise ValueError for "
        "invalid input unless the design says otherwise. Do not write tests."
    )

    def build_prompt(self, ctx):
        prompt = f"""Requirements:
{as_json(list(ctx['requirements'].values()))}

Design:
{as_json(ctx['design'])}
"""
        results = ctx["tests"].get("results")
        if results and results.get("failed"):
            prompt += f"""
Your current code:
{as_json(ctx['code']['files'])}

The latest test run failed. Output (truncated):
{results.get('output', '')[-3000:]}
"""
        prompt += """
Return JSON in exactly this format, with the full content of every file:
{
  "files": {"app/__init__.py": "", "app/module.py": "full source code"}
}"""
        return prompt

    def validate(self, r):
        files = r.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("'files' must be a non-empty object of path -> source")
        for path in files:
            if not path.endswith(".py") or path.startswith(("/", "..")) or ".." in path:
                raise ValueError(f"invalid file path {path!r}")

    def apply(self, r):
        self.store.dispatch("CODE_FILES_WRITTEN", self.name, {"files": r["files"]})


class TestingAgent(BaseAgent):
    name = phase = "testing"
    reads = ("requirements", "design", "code")
    system_prompt = (
        "You are the Testing Agent in PipelineIQ. You write pytest tests that verify the code meets "
        "every acceptance criterion. Test the public interfaces from the design, import from the "
        "'app' package, and use only pytest and the standard library. Name every test after the "
        "requirement it checks, e.g. test_req_1_valid_login, so results trace back to requirements. "
        "Test the requirements, not implementation details."
    )

    def build_prompt(self, ctx):
        return f"""Requirements:
{as_json(list(ctx['requirements'].values()))}

Design:
{as_json(ctx['design'])}

Code under test:
{as_json(ctx['code']['files'])}

Write tests covering every requirement (at least one test per REQ id).
Return JSON in exactly this format:
{{
  "files": {{"tests/test_app.py": "full pytest source"}}
}}"""

    def validate(self, r):
        files = r.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("'files' must be a non-empty object")
        for path in files:
            name = path.rsplit("/", 1)[-1]
            if not (path.startswith("tests/") and name.startswith("test_") and path.endswith(".py")):
                raise ValueError(f"test files must look like tests/test_*.py, got {path!r}")

    def apply(self, r):
        self.store.dispatch("TEST_FILES_WRITTEN", self.name, {"files": r["files"]})
