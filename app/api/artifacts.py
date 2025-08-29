import os, json
from fastapi import APIRouter
from typing import List, Dict
from .utils import ARTIFACTS_DIR

router = APIRouter(tags=["artifacts"])

def _read_metrics_for(basepath: str) -> dict:
    metrics_path = os.path.splitext(basepath)[0] + ".metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@router.get("/artifacts")
def list_artifacts():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(ARTIFACTS_DIR))
    models: List[Dict] = []
    regions = set()
    for f in files:
        full = os.path.join(ARTIFACTS_DIR, f)
        if f.endswith(".pkl"):
            models.append({
                "artifact": f,
                "size_bytes": os.path.getsize(full),
                "metrics": _read_metrics_for(full),
            })
            if "_spx_" in f:
                parts = f[:-4].split("_spx_")
                if len(parts) == 2 and parts[1]:
                    regions.add(parts[1])
    return {
        "dir": ARTIFACTS_DIR,
        "count": len(models),
        "regions": sorted(regions),
        "models": models,
    }
