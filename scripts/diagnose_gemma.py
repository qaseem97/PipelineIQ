"""Find out which Gemma 4 model / setting is failing on the Gemini API.

    python scripts/diagnose_gemma.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.genai import types

from pipelineiq import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
SYSTEM = "You are a helpful assistant."
LONG = "Summarise in one line: " + ("PipelineIQ automates the software development life cycle. " * 200)

checks = []
for model in (config.WORKER_MODEL, config.MANAGER_MODEL):
    checks += [
        (model, "plain", {}),
        (model, "system instruction", {"system_instruction": SYSTEM}),
        (model, "thinking high", {"thinking_config": types.ThinkingConfig(thinking_level="high")}),
        (model, "thinking minimal", {"thinking_config": types.ThinkingConfig(thinking_level="minimal")}),
        (model, "long prompt", {"_prompt": LONG}),
    ]

print(f"Testing {len(checks)} calls...\n")
for model, label, opts in checks:
    prompt = opts.pop("_prompt", "Reply with exactly: OK")
    start = time.time()
    try:
        r = client.models.generate_content(model=model, contents=prompt,
                                           config=types.GenerateContentConfig(**opts))
        result = f"OK    ({(r.text or '').strip()[:40]!r})"
    except Exception as e:  # noqa: BLE001
        result = f"FAIL  {str(e)[:90]}"
    print(f"{model:<22} {label:<20} {time.time() - start:5.1f}s  {result}")
