"""Start the PipelineIQ dashboard:  python scripts/dashboard.py   then open http://localhost:8000"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # keep pipelineiq.db, chroma_db and workspace in the project folder

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"PipelineIQ dashboard: http://localhost:{port}")
    uvicorn.run("pipelineiq.web:app", host=os.getenv("HOST", "127.0.0.1"), port=port)
