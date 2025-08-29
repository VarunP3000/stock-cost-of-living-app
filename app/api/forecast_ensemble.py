from fastapi import APIRouter, Query, HTTPException
from typing import Dict
from .utils import load_model, load_metrics, now_iso, build_latest_feature_row
from app.middleware.logging import log_prediction

router = APIRouter(prefix="/forecast", tags=["forecast"])

MODEL_KEYS = ["ridge", "elasticnet", "gb"]
ARTIFACTS = {
    "ridge": "ridge_spx_v1.pkl",
    "elasticnet": "elasticnet_spx_v1.pkl",
    "gb": "gb_spx_v1.pkl",
}

def parse_weights(w: str | None) -> Dict[str, float]:
    if not w:
        return {}
    out = {}
    for part in w.split(","):
        k, v = part.split(":")
        k = k.strip().lower()
        if k not in MODEL_KEYS:
            raise HTTPException(400, f"Unknown model '{k}'. Use {MODEL_KEYS}.")
        out[k] = float(v)
    return out

@router.get("/ensemble")
def forecast_ensemble(weights: str | None = Query(None, description="e.g. ridge:.4,elasticnet:.3,gb:.3")):
    X = build_latest_feature_row()  # 1 x n_features

    preds = {}
    metas = {}
    for k in MODEL_KEYS:
        model = load_model(ARTIFACTS[k])
        yhat = float(model.predict(X)[0])
        preds[k] = yhat
        metas[k] = {
            "artifact": ARTIFACTS[k],
            "metrics": load_metrics(ARTIFACTS[k]),
        }

    w_map = parse_weights(weights)
    if any(k not in MODEL_KEYS for k in w_map.keys()):
        raise HTTPException(400, f"weights keys must be subset of {MODEL_KEYS}")
    if not w_map:
        w_map = {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}
    total = sum(w_map.values())
    if total <= 0:
        raise HTTPException(400, "weights must sum to > 0")
    w_map = {k: v / total for k, v in w_map.items()}

    ensemble_pred = float(sum(w_map[k] * preds[k] for k in MODEL_KEYS))

    resp = {
        "asof": now_iso(),
        "prediction": ensemble_pred,
        "components": preds,
        "weights_used": w_map,
        "metadata": metas,
        "feature_order": "training-aligned",
    }

    try:
        log_prediction("/forecast/ensemble", resp, features=X[0].tolist())
    finally:
        return resp
