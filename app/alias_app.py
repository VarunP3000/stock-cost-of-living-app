# app/alias_app.py
# Adds /forecast/ensemble and /forecast/quantiles to your existing FastAPI app
# by reading artifacts created earlier:
#   - data/ensemble_weights.json
#   - data/forecast_scenarios.json
#
# It leaves all your existing routes (e.g., /correlations) intact.

from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path
import json
from fastapi import HTTPException, Query

# Import your existing FastAPI app instance
try:
    from app.main import app  # type: ignore
except Exception:
    from fastapi import FastAPI
    app = FastAPI(title="Alias App")
    # If this triggered, you can later mount your routers here as needed.

DATA_SCEN = Path("data/forecast_scenarios.json")
DATA_WTS  = Path("data/ensemble_weights.json")

def _load_scenarios() -> Dict:
    if not DATA_SCEN.exists():
        raise HTTPException(status_code=404, detail=f"{DATA_SCEN} not found. Run scripts/learn_weights_and_scenarios.py")
    with DATA_SCEN.open() as f:
        return json.load(f)

def _parse_weights_arg(weights: Optional[str]) -> Dict[str, float]:
    # "ridge:0.4,elasticnet:0.3,gb:0.3"
    out: Dict[str, float] = {}
    if not weights:
        return out
    for pair in weights.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out

def _normalize_positive(w: Dict[str, float]) -> Dict[str, float]:
    # keep only positive entries and renormalize to sum=1
    w = {k: float(v) for k, v in w.items() if v and v > 0}
    s = sum(w.values())
    if s <= 1e-12:
        return {}
    return {k: v / s for k, v in w.items()}

@app.get("/forecast/ensemble")
def forecast_ensemble(
    weights: Optional[str] = Query(None, description="Custom weights like 'ridge:0.4,elasticnet:0.3,gb:0.3'"),
    scenario: str = Query("baseline", pattern="^(baseline|optimistic|pessimistic)$"),
    horizon: int = Query(1, ge=1, le=60, description="Months ahead (1 = next month)")
):
    """
    Returns a single-horizon ensemble prediction computed from scenario artifacts.
    Shape is simple and friendly to your metrics panel: {model,prediction,asof,horizon,scenario}.
    """
    d = _load_scenarios()
    fut_dates = d.get("future_dates", [])
    if not fut_dates:
        raise HTTPException(status_code=500, detail="future_dates missing in scenarios artifact")
    if scenario not in d.get("scenarios", {}):
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario}' not found")

    sc = d["scenarios"][scenario]
    # Base members at this horizon (arrays are 1..H; our index is horizon-1)
    idx = horizon - 1
    try:
        base_vals = {
            "ridge": float(sc["ridge"][idx]),
            "elasticnet": float(sc["elasticnet"][idx]),
            "gb": float(sc["gb"][idx]),
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Base model arrays missing or too short")

    # Decide weights: request override > artifact > equal
    req_w = _normalize_positive(_parse_weights_arg(weights))
    if req_w:
        w = req_w
    else:
        w = d.get("weights", {})
        w = _normalize_positive({k: float(v) for k, v in w.items()}) or {"ridge": 1/3, "elasticnet": 1/3, "gb": 1/3}

    # Renormalize over models that exist
    keys = [k for k in base_vals.keys() if w.get(k, 0.0) > 0]
    if not keys:
        keys = list(base_vals.keys())
        w = {k: 1/3 for k in keys}
    s = sum(w[k] for k in keys)
    pred = sum((w[k]/s) * base_vals[k] for k in keys)

    return {
        "model": "ensemble",
        "prediction": float(pred),
        "asof": fut_dates[idx],
        "horizon": horizon,
        "scenario": scenario,
    }

@app.get("/forecast/quantiles")
def forecast_quantiles(
    scenario: str = Query("baseline", pattern="^(baseline|optimistic|pessimistic)$"),
    horizon: int = Query(1, ge=1, le=60)
):
    """
    Returns quantile band for the requested horizon.
    Shape includes p10/p50/p90 plus arrays for flexibility.
    """
    d = _load_scenarios()
    fut_dates = d.get("future_dates", [])
    if not fut_dates:
        raise HTTPException(status_code=500, detail="future_dates missing in scenarios artifact")
    if scenario not in d.get("scenarios", {}):
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario}' not found")

    sc = d["scenarios"][scenario]
    idx = horizon - 1
    try:
        p10_arr = [float(x) for x in sc["p10"]]
        p90_arr = [float(x) for x in sc["p90"]]
    except Exception:
        raise HTTPException(status_code=500, detail="p10/p90 arrays missing")

    if idx >= len(p10_arr) or idx >= len(p90_arr):
        raise HTTPException(status_code=400, detail="Requested horizon exceeds available range")

    # crude p50 as midpoint if not present
    p50_arr = [0.5*(a+b) for a, b in zip(p10_arr, p90_arr)]

    return {
        "model": "quantiles",
        "horizons": list(range(1, len(p10_arr)+1)),
        "future_dates": fut_dates,
        "p10": p10_arr,
        "p50": p50_arr,
        "p90": p90_arr,
        "point": {  # convenience for horizon
            "asof": fut_dates[idx],
            "horizon": horizon,
            "p10": float(p10_arr[idx]),
            "p50": float(p50_arr[idx]),
            "p90": float(p90_arr[idx]),
        },
        "scenario": scenario,
    }
