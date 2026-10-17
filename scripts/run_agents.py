"""Step 2 demo: run the four phase agents once, in order, on one idea.

    python scripts/run_agents.py "Build a to-do list manager with priorities and due dates"

(The Manager's review-and-rework loop is added in Step 3.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelineiq import config
from pipelineiq.agents import AGENTS, PHASE_ORDER
from pipelineiq.llm import get_llm
from pipelineiq.store import Store
from pipelineiq.test_runner import run_tests

idea = " ".join(sys.argv[1:]) or "Build a to-do list manager with priorities and due dates"
llm = get_llm()
store = Store.create(name=idea[:60], idea=idea)
store.subscribe(lambda a, s: print(f"  -> action {a['type']} by {a['agent']}"))
print(f"Project {store.project_id}: {idea}\n")

for phase in PHASE_ORDER:
    task_id = f"{phase.upper()}-1"
    store.dispatch("PHASE_CHANGED", "orchestrator", {"phase": phase})
    store.dispatch("TASK_CREATED", "orchestrator", {"task_id": task_id, "phase": phase, "agent": phase})
    print(f"[{phase}] agent working on Gemma 4 ({config.WORKER_MODEL})...")
    AGENTS[phase](store, llm).run(task_id)

state = store.state
print("\nRequirements:")
for r in state["requirements"].values():
    print(f"  {r['id']}: {r['title']}")
print("\nDesign:", state["design"].get("summary", ""))
print("\nCode files:", ", ".join(state["code"]["files"]))
print("Test files:", ", ".join(state["tests"]["files"]))

print("\nRunning generated tests...")
results = run_tests(store)
print(f"Tests: {results['passed']} passed, {results['failed']} failed (of {results['total']})")
for c in results["cases"]:
    print(f"  [{c['outcome'].upper():7}] {c['name']}")
print(f"\nGenerated project is in workspace/{store.project_id}/")
