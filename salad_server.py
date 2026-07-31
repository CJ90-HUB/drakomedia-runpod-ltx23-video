from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException

from handler import handler
from salad_assets import asset_progress_snapshot, ensure_model_assets


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
app = FastAPI(title="DrakoMedia SaladCloud LTX-2.3")


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/status")
def status() -> dict[str, Any]:
    return asset_progress_snapshot()


@app.post("/process")
def process(job_data: dict[str, Any]) -> dict[str, Any]:
    payload = job_data.get("input", job_data)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="La solicitud no contiene un objeto de entrada.")
    started = time.monotonic()
    try:
        operation = str(payload.get("operation", "generate_video")).strip()
        required_roots = {"gemma"} if operation == "analyze_motion" else {"gemma", "ltx"}
        print(
            f"DRAKO_SALAD job_start operation={operation} roots={','.join(sorted(required_roots))}",
            flush=True,
        )
        cache = ensure_model_assets(
            payload.get("model_manifest"),
            required_roots=required_roots,
            operation=operation,
        )
        print(f"DRAKO_SALAD inference_start operation={operation}", flush=True)
        result = handler({"input": payload})
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error", "LTX-2.3 no completó el vídeo.")))
        result.update(
            {
                "provider": "saladcloud",
                "gpu_class": "rtx5090",
                "model_cache": cache,
                "worker_total_ms": round((time.monotonic() - started) * 1000),
            }
        )
        print(f"DRAKO_SALAD job_complete operation={operation}", flush=True)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        print(f"DrakoMedia SaladCloud · error {type(exc).__name__}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc)[:500]) from exc
