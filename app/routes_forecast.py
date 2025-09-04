# app/routes_forecast.py
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime
from pathlib import Path
import csv
import pandas as pd
import numpy as np

router = APIRouter(prefix="/forecast", tags=["forecast"])

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

BASE_MODELS = ["ridge", "elasticnet", "gb"]
DEFAULT_WEIGHTS = {"ridge": 0.3, "elasticnet": 0.3, "gb": 0.4}  # sum ≈ 1.0

def _norm_month_str(s: str) -> str | None:
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_period("M").to_timestamp().strftime("%Y-%m-%d")
    except Exception:
        return None

def _fnum(x):
    try:
        if x in ("", None):
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def _ym_key(s: str) -> tuple[int, int]:
    if not s:
        return (0, 0)
    y, m = s[:7].split("-")
    return (int(y), int(m))

def _parse_weights_param(weights: str | None, models: list[str]) -> list[float]:
    if not weights:
        w = {k: DEFAULT_WEIGHTS.get(k, 0.0) for k in models}
    else:
        try:
            parts = [p.strip() for p in weights.split(",") if p.strip()]
            w = {}
            for part in parts:
                k, v = [x.strip() for x in part.split(":")]
                if k in models:
                    w[k] = float(v)
            # fill missing with 0
            for k in models:
                w.setdefault(k, 0.0)
        except Exception:
            w = {k: DEFAULT_WEIGHTS.get(k, 0.0) for k in models}
    s = sum(w.values())
    if s > 0 and not np.isclose(s, 1.0):
        w = {k: (v / s) for k, v in w.items()}
    return [w[k] for k in models]

def _load_backtest_csv(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing {path}. Run backtest_export_series.py first.")
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = [c.lower() for c in (rdr.fieldnames or [])]
        date_col = next((c for c in ("month", "asof", "date") if c in cols), None)
        if not date_col:
            raise HTTPException(status_code=500, detail=f"{path.name} missing date column (month/asof/date).")
        for r in rdr:
            lr = {k.lower(): v for k, v in r.items()}
            m = _norm_month_str(lr.get(date_col))
            if not m:
                continue
            rows.append({
                "month": m,
                "prediction": _fnum(lr.get("prediction")),
                "actual": _fnum(lr.get("actual")),
                "p10": _fnum(lr.get("p10")),
                "p90": _fnum(lr.get("p90")),
            })
    rows.sort(key=lambda r: _ym_key(r["month"]))
    return rows

def _load_future_csv(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing {path}. Precompute your future path and save it there.")
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = [c.lower() for c in (rdr.fieldnames or [])]
        date_col = next((c for c in ("month", "asof", "date") if c in cols), None)
        if not date_col:
            raise HTTPException(status_code=500, detail=f"{path.name} missing date column (month/asof/date).")
        if "prediction" not in cols:
            raise HTTPException(status_code=500, detail=f"{path.name} missing 'prediction' column.")
        for r in rdr:
            lr = {k.lower(): v for k, v in r.items()}
            m = _norm_month_str(lr.get(date_col))
            if not m:
                continue
            rows.append({"month": m, "prediction": _fnum(lr.get("prediction"))})
    rows.sort(key=lambda r: _ym_key(r["month"]))
    return rows

def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and np.isfinite(v)]
    if not pairs:
        return None
    vs, ws = zip(*pairs)
    s = float(np.dot(vs, ws))
    wsum = float(np.sum(ws))
    if wsum <= 0:
        return None
    return s / wsum

# -------- HISTORY (single model or ensemble) --------
@router.get("/history")
def history(
    model: str = Query("ensemble", pattern="^(ensemble|ridge|elasticnet|gb)$"),
    start: str | None = Query(None, description="YYYY-MM"),
    end: str | None = Query(None, description="YYYY-MM"),
    weights: str | None = Query(None, description="ridge:0.3,elasticnet:0.3,gb:0.4"),
    window_months: int | None = Query(None, ge=6, le=240),
):
    # ENSEMBLE (preferred: blend base models that exist; fallback: precomputed ensemble file)
    if model == "ensemble":
        existing_models: list[str] = []
        model_rows: dict[str, list[dict]] = {}
        for m in BASE_MODELS:
            p = REPORTS / f"backtest_series_forecast_{m}.csv"
            if p.exists():
                model_rows[m] = _load_backtest_csv(p)
                existing_models.append(m)

        if not existing_models:
            # fallback to precomputed ensemble file if available
            p = REPORTS / "backtest_series_forecast_ensemble.csv"
            if not p.exists():
                return {
                    "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dates": [], "prediction": [], "actual": [], "p10": [], "p90": [],
                    "models": [], "weights": [],
                }
            rows = _load_backtest_csv(p)
            # filter start/end
            if start:
                rows = [r for r in rows if _ym_key(r["month"]) >= _ym_key(start + "-01")]
            if end:
                rows = [r for r in rows if _ym_key(r["month"]) <= _ym_key(end + "-01")]
            dates = [r["month"] for r in rows]
            pred  = [r["prediction"] for r in rows]
            act   = [r["actual"] for r in rows]
            p10   = [r["p10"] for r in rows]
            p90   = [r["p90"] for r in rows]
            # window trim
            if window_months is not None and len(dates) > window_months:
                dates, pred, act, p10, p90 = dates[-window_months:], pred[-window_months:], act[-window_months:], p10[-window_months:], p90[-window_months:]
            return {
                "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dates": dates, "prediction": pred, "actual": act, "p10": p10, "p90": p90,
                "models": ["ensemble"], "weights": [1.0],
            }

        # blend existing base models
        ws = _parse_weights_param(weights, existing_models)
        # union of months across existing models
        all_months = sorted(set().union(*[set(r["month"] for r in rows) for rows in model_rows.values()]))

        # filter window
        months = all_months
        if start:
            months = [d for d in months if _ym_key(d) >= _ym_key(start + "-01")]
        if end:
            months = [d for d in months if _ym_key(d) <= _ym_key(end + "-01")]
        if not months:
            raise HTTPException(status_code=404, detail="No rows in selected window.")

        dates, pred, act, p10, p90 = [], [], [], [], []
        for d in months:
            dates.append(d)
            preds, acts, p10s, p90s = [], [], [], []
            for m in existing_models:
                row = next((r for r in model_rows[m] if r["month"] == d), None)
                preds.append(row["prediction"] if row else None)
                acts.append(row["actual"] if row else None)
                p10s.append(row["p10"] if row else None)
                p90s.append(row["p90"] if row else None)
            pred.append(_weighted_mean(preds, ws))
            act.append(_weighted_mean(acts, [1.0]*len(existing_models)))  # simple mean of available actuals
            p10.append(_weighted_mean(p10s, ws))
            p90.append(_weighted_mean(p90s, ws))

        # trailing window
        if window_months is not None and len(dates) > window_months:
            dates, pred, act, p10, p90 = dates[-window_months:], pred[-window_months:], act[-window_months:], p10[-window_months:], p90[-window_months:]

        return {
            "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dates": dates, "prediction": pred, "actual": act, "p10": p10, "p90": p90,
            "models": existing_models, "weights": ws,
        }

    # SINGLE MODEL
    p = REPORTS / f"backtest_series_forecast_{model}.csv"
    rows = _load_backtest_csv(p)
    if start:
        rows = [r for r in rows if _ym_key(r["month"]) >= _ym_key(start + "-01")]
    if end:
        rows = [r for r in rows if _ym_key(r["month"]) <= _ym_key(end + "-01")]
    if not rows:
        raise HTTPException(status_code=404, detail="No rows in selected window.")
    dates = [r["month"] for r in rows]
    pred  = [r["prediction"] for r in rows]
    act   = [r["actual"] for r in rows]
    p10   = [r["p10"] for r in rows]
    p90   = [r["p90"] for r in rows]
    if window_months is not None and len(dates) > window_months:
        dates, pred, act, p10, p90 = dates[-window_months:], pred[-window_months:], act[-window_months:], p10[-window_months:], p90[-window_months:]
    return {
        "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dates": dates, "prediction": pred, "actual": act, "p10": p10, "p90": p90,
    }

# -------- FUTURE (single model or ensemble) --------
@router.get("/future")
def future(
    model: str = Query("ensemble", pattern="^(ensemble|ridge|elasticnet|gb)$"),
    horizon: int = Query(6, ge=1, le=24),
    weights: str | None = Query(None, description="ridge:0.3,elasticnet:0.3,gb:0.4"),
):
    # ENSEMBLE
    if model == "ensemble":
        existing_models: list[str] = []
        futures: dict[str, dict[str, float | None]] = {}
        for m in BASE_MODELS:
            p = REPORTS / f"future_series_{m}.csv"
            if p.exists():
                rows = _load_future_csv(p)
                futures[m] = {r["month"]: r["prediction"] for r in rows}
                existing_models.append(m)

        if not existing_models:
            # Graceful placeholder: next months with null predictions
            from dateutil.relativedelta import relativedelta
            start = (datetime.utcnow().replace(day=1) + relativedelta(months=1))
            months = [(start + relativedelta(months=i)).strftime("%Y-%m-%d") for i in range(horizon)]
            return {
                "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "future_dates": months,
                "future_prediction": [None] * len(months),
                "models": [], "weights": [],
            }

        W = _parse_weights_param(weights, existing_models)
        months = sorted(set().union(*[set(d.keys()) for d in futures.values()]))[:horizon]
        yhat = []
        for d in months:
            vals = [futures[m].get(d) for m in existing_models]
            yhat.append(_weighted_mean(vals, W))
        return {
            "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "future_dates": months,
            "future_prediction": yhat,
            "models": existing_models, "weights": list(W),
        }

    # SINGLE MODEL
    p = REPORTS / f"future_series_{model}.csv"
    if not p.exists():
        # same graceful placeholder for missing single-model future
        from dateutil.relativedelta import relativedelta
        start = (datetime.utcnow().replace(day=1) + relativedelta(months=1))
        months = [(start + relativedelta(months=i)).strftime("%Y-%m-%d") for i in range(horizon)]
        return {
            "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "future_dates": months,
            "future_prediction": [None] * len(months),
        }
    rows = _load_future_csv(p)[:horizon]
    return {
        "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "future_dates": [r["month"] for r in rows],
        "future_prediction": [r["prediction"] for r in rows],
    }
