"""Runs the generated tests against the generated code (a deterministic quality check).

Files from the shared state are written into workspace/<project_id>/ and pytest runs
there in a subprocess with a timeout. Results are dispatched to the store.

Note: this executes AI-generated code on the machine running PipelineIQ. Keep it in a
throwaway environment (container or VM) when deployed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .store import Store

WORKSPACE = Path("workspace")


def materialize(store: Store) -> Path:
    state = store.state
    root = WORKSPACE / store.project_id
    if root.exists():
        shutil.rmtree(root)
    for path, source in {**state["code"]["files"], **state["tests"]["files"]}.items():
        target = (root / path).resolve()
        if root.resolve() not in target.parents:
            raise ValueError(f"Refusing to write outside the workspace: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    (root / "app").mkdir(exist_ok=True)
    (root / "app" / "__init__.py").touch()
    (root / "conftest.py").write_text("", encoding="utf-8")  # makes 'app' importable from tests
    # Own pytest.ini so pytest does not pick up PipelineIQ's config from a parent folder.
    (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n", encoding="utf-8")
    return root


def run_tests(store: Store, timeout: int = 60) -> dict:
    root = materialize(store)
    report = root / "report.xml"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report.name}", "tests"],
            cwd=root, capture_output=True, text=True, timeout=timeout)
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        output = f"Test run timed out after {timeout}s"

    cases = []
    if report.exists():
        for tc in ET.parse(report).iter("testcase"):
            failed = tc.find("failure") is not None or tc.find("error") is not None
            skipped = tc.find("skipped") is not None
            cases.append({"name": tc.get("name"), "outcome": "skipped" if skipped else ("failed" if failed else "passed")})

    passed = sum(c["outcome"] == "passed" for c in cases)
    failed = sum(c["outcome"] == "failed" for c in cases)
    results = {
        "passed": passed,
        "failed": failed if cases else 1,   # no report at all (e.g. import/syntax error) counts as failure
        "total": len(cases),
        "cases": cases,
        "output": output[-6000:],
    }
    store.dispatch("TEST_RESULTS_RECORDED", "test_runner", results)
    return results
