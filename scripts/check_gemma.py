"""Step 1 check: confirms your API key works and both Gemma 4 models respond."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelineiq import config
from pipelineiq.llm import GemmaClient

client = GemmaClient()
for model in (config.WORKER_MODEL, config.MANAGER_MODEL):
    reply = client.generate("Reply with exactly: PipelineIQ is online.", model=model)
    print(f"[{model}] {reply.strip()}")
print("\nGemma 4 is working. Step 1 complete.")
