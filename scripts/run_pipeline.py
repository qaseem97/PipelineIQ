"""Step 3: run the full pipeline with the Manager's quality gates and rework loop.

    python scripts/run_pipeline.py "Build a to-do list manager with priorities and due dates"
    python scripts/run_pipeline.py "Build a library book lending system" --image sketch.jpg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelineiq import config
from pipelineiq.llm import get_llm
from pipelineiq.orchestrator import Orchestrator
from pipelineiq.store import Store

parser = argparse.ArgumentParser()
parser.add_argument("idea", nargs="*", default=["Build a to-do list manager with priorities and due dates"])
parser.add_argument("--image", action="append", default=[], help="optional sketch/whiteboard photo (jpg)")
args = parser.parse_args()
idea = " ".join(args.idea)
images = [Path(p).read_bytes() for p in args.image]

ICON = {"APPROVE": "APPROVED", "REWORK": "REWORK  ", "ESCALATE": "ESCALATE"}


def show(kind, data):
    if kind == "knowledge":
        added = sum(data["added"].values())
        print(f"RAG knowledge bases: {data['stats']}  (newly embedded chunks: {added})")
    elif kind == "agent_done" and data.get("retrieved"):
        print("  RAG retrieved: " + ", ".join(f"{h['source']} ({h['score']})" for h in data["retrieved"]))
    elif kind == "agent_start":
        print(f"\n[{data['phase']}] attempt {data['attempt']} - agent working ({config.WORKER_MODEL})...")
    elif kind == "recheck":
        print(f"\n[{data['phase']}] output unchanged - re-checking after upstream fix...")
    elif kind == "review":
        score = f" score={data['score']}" if data["score"] is not None else ""
        print(f"  Manager ({config.MANAGER_MODEL}): {ICON[data['decision']]}{score}")
        for c in data["checks"]:
            mark = "ok " if c["result"] == "PASS" else "XX "
            reason = f" - {c['reason']}" if c.get("reason") else ""
            print(f"     {mark}{c['name']}{reason}")
        if data["decision"] != "APPROVE" and data["feedback"]:
            target = f" -> back to {data['route_to']}" if data.get("route_to") else ""
            print(f"  Feedback{target}:")
            for line in data["feedback"].splitlines():
                print(f"     {line}")
    elif kind == "escalated":
        print(f"\nESCALATED in {data['phase']}: {data['feedback']}")


store = Store.create(name=idea[:60], idea=idea)
print(f"Project {store.project_id}: {idea}")
def retry(info):
    print(f"  ! Gemini API error on {info['model']}: {info['error'][:90]}")
    print(f"    retrying in {info['wait']}s with {info['next_model']}"
          f"{'' if info['thinking'] else ' (thinking off)'}...")


orch = Orchestrator(store, get_llm(on_retry=retry), on_event=show)
outcome = orch.run(images=images or None)

print("\n" + "=" * 60)
print(f"Outcome: {outcome.upper()}")
s = store.state
res = s["tests"]["results"] or {}
print(f"Requirements: {len(s['requirements'])} | Code files: {len(s['code']['files'])} | "
      f"Tests: {res.get('passed', 0)}/{res.get('total', 0)} passing")
print("\nProcess log:")
for e in orch.log.entries():
    print(f"  #{e['id']:<3} {e['phase']:<13} attempt {e['attempt']}  {e['status']}  -> {e['decision']}")
print(f"\nGenerated project: workspace/{store.project_id}/")
