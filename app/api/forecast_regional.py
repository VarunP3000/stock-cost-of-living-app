from fastapi import APIRouter, Query, HTTPException
from .utils import load_model, load_metrics, now_iso, build_latest_feature_row, try_get_coefficients
from app.middleware.logging import log_prediction

router = APIRouter(prefix="/forecast", tags=["forecast"])

VALID = {"americas", "emea", "apac"}

@router.get("/regional")
def forecast_regional(region: str = Query(..., pattern="^(americas|emea|apac)$")):
    r = region.lower()
    if r not in VALID:
        raise HTTPException(400, f"region must be one of {sorted(VALID)}")

    artifact = f"ridge_spx_{r}.pkl"
    model = load_model(artifact)
    X = build_latest_feature_row(region=r)
    yhat = float(model.predict(X)[0])

    resp = {
        "asof": now_iso(),
        "region": r,
        "prediction": yhat,
        "metadata": {
            "artifact": artifact,
            "metrics": load_metrics(artifact),
            "coefficients": try_get_coefficients(model),
        },
        "feature_order": "training-aligned",
    }

    try:
        log_prediction("/forecast/regional", resp, features=X[0].tolist())
    finally:
        return resp
