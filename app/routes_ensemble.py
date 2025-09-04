# app/routes_ensemble.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone


router = APIRouter(prefix="/ensemble", tags=["ensemble"])

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
PRED_TIDY = DATA / "predictions_tidy.csv"
SPX_ACTUALS = DATA / "spx_actuals.csv"

# -------------------- utils --------------------
def _to_month_start(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

def _nan_to_none(x: Any) -> Any:
    if isinstance(x, (float, np.floating)):
        if math.isfinite(float(x)):
            return float(x)
        return None
    if x is None:
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.ndarray, list, tuple)):
        return [ _nan_to_none(v) for v in list(x) ]
    if isinstance(x, dict):
        return { k: _nan_to_none(v) for k, v in x.items() }
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x

def _read_csv(path: Path) -> pd.DataFrame:
    # tolerant CSV reader
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

def _detect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

# -------------------- building tidy predictions if needed --------------------
def _build_predictions_from_reports() -> pd.DataFrame:
    files = sorted(REPORTS.glob("backtest_series_forecast_*.csv"))
    if not files:
        raise HTTPException(
            404,
            f"Missing {PRED_TIDY} and no reports/backtest_series_forecast_*.csv files found to build it."
        )

    rows = []
    for fp in files:
        df = _read_csv(fp)
        if df is None or df.empty:
            continue

        date_col = _detect_col(
            list(df.columns),
            ["asof", "month", "date", "ds", "Date", "timestamp", "period", "time"]
        )
        pred_col = _detect_col(
            list(df.columns),
            [
                "yhat", "prediction", "pred", "y_pred", "y_pred_mean",
                "forecast", "mean", "median", "value", "yhat_mean"
            ]
        )
        if not date_col or not pred_col:
            # skip unusable backtest file
            continue

        df["asof"] = _to_month_start(df[date_col])
        df["yhat"] = pd.to_numeric(df[pred_col], errors="coerce")

        # model name from filename suffix
        # e.g. backtest_series_forecast_ridge.csv -> model="ridge"
        name = fp.stem.replace("backtest_series_forecast_", "")
        df["model"] = str(name)

        rows.append(df[["asof", "model", "yhat"]])

    if not rows:
        raise HTTPException(
            400,
            "Could not infer predictions from reports/*.csv (no usable date/pred columns)."
        )

    tidy = pd.concat(rows, ignore_index=True)
    tidy = tidy.dropna(subset=["asof"]).sort_values(["asof", "model"])
    # cache to data/
    try:
        tidy.to_csv(PRED_TIDY, index=False)
    except Exception:
        # ignore write errors; still return df
        pass
    return tidy

def _load_predictions_tidy() -> pd.DataFrame:
    if PRED_TIDY.exists():
        df = _read_csv(PRED_TIDY)
        if df is None or df.empty:
            # fall through to builder
            return _build_predictions_from_reports()
        # Flexible detection
        date_col = _detect_col(list(df.columns), ["asof","month","date","ds","Date","timestamp","period","time"])
        model_col = _detect_col(list(df.columns), ["model","name","estimator","model_name","algo","algorithm"])
        yhat_col  = _detect_col(list(df.columns), ["yhat","prediction","pred","y_pred","value","forecast","mean","median","yhat_mean"])

        if not date_col:
            return _build_predictions_from_reports()
        if not model_col:
            return _build_predictions_from_reports()
        if not yhat_col:
            return _build_predictions_from_reports()

        df["asof"] = _to_month_start(df[date_col])
        df["model"] = df[model_col].astype(str)
        df["yhat"] = pd.to_numeric(df[yhat_col], errors="coerce")
        df = df[["asof","model","yhat"]].dropna(subset=["asof"])
        return df.sort_values(["asof","model"])

    # No tidy file → build from reports
    return _build_predictions_from_reports()

def _load_actuals() -> pd.DataFrame:
    if not SPX_ACTUALS.exists():
        raise HTTPException(404, f"Missing {SPX_ACTUALS}")
    adf = _read_csv(SPX_ACTUALS)
    date_col = _detect_col(list(adf.columns), ["asof","month","date","ds","Date","timestamp","period","time"])
    if not date_col:
        raise HTTPException(400, "spx_actuals.csv needs a date column (asof/month/date/ds/Date/timestamp/period/time)")
    if "actual" not in adf.columns:
        alt_val = _detect_col(list(adf.columns), ["actual_return","spx_ret","return","y","value"])
        if alt_val:
            adf = adf.rename(columns={alt_val:"actual"})
    if "actual" not in adf.columns:
        raise HTTPException(400, "spx_actuals.csv must have 'actual' (or one of: actual_return/spx_ret/return/y/value)")
    adf["asof"] = _to_month_start(adf[date_col])
    adf["actual"] = pd.to_numeric(adf["actual"], errors="coerce")
    adf = adf[["asof","actual"]].dropna(subset=["asof"]).drop_duplicates("asof")
    return adf.sort_values("asof")

def _normalize_weights(models: List[str], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not models:
        return {}
    if not weights:
        return {m: 1.0/len(models) for m in models}
    w = {m: float(weights.get(m, 0.0)) for m in models}
    s = sum(w.values())
    if s <= 0:
        return {m: 1.0/len(models) for m in models}
    return {m: v/s for m, v in w.items()}

# -------------------- schemas --------------------
class PastRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    weights: Optional[Dict[str, float]] = None

class PastResponse(BaseModel):
    asof: str
    dates: List[str]
    prediction: List[Optional[float]]
    actual: List[Optional[float]]
    p10: List[Optional[float]]
    p90: List[Optional[float]]
    models: List[str]
    weights: List[float]

class FutureRequest(BaseModel):
    horizon: int = 6
    weights: Optional[Dict[str, float]] = None

class FutureResponse(BaseModel):
    asof: str
    future_dates: List[str]
    future_prediction: List[Optional[float]]
    p10: List[Optional[float]]
    p90: List[Optional[float]]
    models: List[str]
    weights: List[float]

# -------------------- endpoints --------------------
@router.post("/past", response_model=PastResponse)
def past(req: PastRequest):
    pred = _load_predictions_tidy()
    acts = _load_actuals()

    if req.start:
        pred = pred[pred["asof"] >= pd.to_datetime(req.start)]
        acts = acts[acts["asof"] >= pd.to_datetime(req.start)]
    if req.end:
        pred = pred[pred["asof"] <= pd.to_datetime(req.end)]
        acts = acts[acts["asof"] <= pd.to_datetime(req.end)]

    if pred.empty:
        raise HTTPException(404, "No predictions in requested range")

    models = sorted(pred["model"].dropna().astype(str).unique().tolist())
    if not models:
        # We built tidy, but nothing usable
        raise HTTPException(404, "No model names could be inferred from predictions input")

    wide = pred.pivot_table(index="asof", columns="model", values="yhat", aggfunc="mean").sort_index()

    base_weights = _normalize_weights(models, req.weights)
    ens_vals, std_vals = [], []
    for _, row in wide.iterrows():
        row_models = [m for m in models if pd.notna(row.get(m))]
        if not row_models:
            ens_vals.append(np.nan)
            std_vals.append(np.nan)
            continue
        w_sub = _normalize_weights(row_models, {m: base_weights.get(m, 0.0) for m in row_models})
        preds = np.array([float(row[m]) for m in row_models], dtype=float)
        wvec = np.array([w_sub[m] for m in row_models], dtype=float)
        ens_vals.append(float(np.dot(preds, wvec)))
        std_vals.append(float(np.std(preds)) if len(preds) > 1 else 0.0)

    wide = wide.assign(ensemble=ens_vals, model_std=std_vals)
    acts = acts.set_index("asof")
    wide = wide.join(acts["actual"], how="left")

    dates = [d.strftime("%Y-%m-%d") for d in wide.index]
    prediction = [None if pd.isna(v) else float(v) for v in wide["ensemble"].tolist()]
    actual = [None if "actual" not in wide or pd.isna(v) else float(v) for v in wide.get("actual", pd.Series(index=wide.index, dtype=float)).tolist()]
    std = wide["model_std"].fillna(0.0).astype(float).tolist()
    p10, p90 = [], []
    for y, s in zip(prediction, std):
        if y is None:
            p10.append(None)
            p90.append(None)
        else:
            p10.append(float(y - s))
            p90.append(float(y + s))

    out = {
        "asof": datetime.utcnow().isoformat(),
        "dates": dates,
        "prediction": prediction,
        "actual": actual,
        "p10": p10,
        "p90": p90,
        "models": models,
        "weights": [base_weights[m] for m in models],
    }
    return _nan_to_none(out)

@router.post("/future", response_model=FutureResponse)
def future(req: FutureRequest):
    pred = _load_predictions_tidy()
    acts = _load_actuals()

    models = sorted(pred["model"].dropna().astype(str).unique().tolist())
    if not models:
        raise HTTPException(404, "No model names could be inferred from predictions input")
    base_weights = _normalize_weights(models, req.weights)

    last_actual = acts["asof"].max()
    if pd.isna(last_actual):
        raise HTTPException(404, "No actuals found")

    up_to = pred[pred["asof"] <= last_actual]
    if up_to.empty:
        up_to = pred[pred["asof"] == pred["asof"].max()]
    latest = up_to.groupby("model")["yhat"].last()

    row_models = [m for m in models if m in latest.index and pd.notna(latest[m])]
    if not row_models:
        raise HTTPException(404, "No model predictions available at the latest date")
    w_sub = _normalize_weights(row_models, {m: base_weights.get(m, 0.0) for m in row_models})
    preds = np.array([float(latest[m]) for m in row_models], dtype=float)
    wvec = np.array([w_sub[m] for m in row_models], dtype=float)
    yhat_last = float(np.dot(preds, wvec))
    disagree = float(np.std(preds)) if len(preds) > 1 else 0.0

    # quick error cushion from last 24 months
    try:
        wide = pred.pivot_table(index="asof", columns="model", values="yhat", aggfunc="mean").sort_index()
        wide = wide[wide.index.isin(acts["asof"])].tail(24)
        ens_hist, act_hist = [], []
        aw = acts.set_index("asof")["actual"]
        for ts, row in wide.iterrows():
            rm = [m for m in models if pd.notna(row.get(m))]
            if not rm:
                continue
            ws = _normalize_weights(rm, {m: base_weights.get(m, 0.0) for m in rm})
            pr = np.array([float(row[m]) for m in rm], dtype=float)
            wv = np.array([ws[m] for m in rm], dtype=float)
            ens_hist.append(float(np.dot(pr, wv)))
            act_hist.append(float(aw.get(ts)) if not pd.isna(aw.get(ts)) else np.nan)
        mask = ~np.isnan(ens_hist) & ~np.isnan(act_hist)
        mae = float(np.mean(np.abs(np.array(ens_hist)[mask] - np.array(act_hist)[mask]))) if mask.any() else 0.0
    except Exception:
        mae = 0.0

    start_future = (last_actual + pd.offsets.MonthBegin(1)).normalize()
    future_idx = pd.date_range(start_future, periods=max(1, req.horizon), freq="MS")
    future_dates = [d.strftime("%Y-%m-%d") for d in future_idx]

    band = disagree + mae
    future_prediction = [yhat_last for _ in future_dates]
    p10 = [yhat_last - band for _ in future_dates]
    p90 = [yhat_last + band for _ in future_dates]

    out = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "future_dates": future_dates,
        "future_prediction": future_prediction,
        "p10": p10,
        "p90": p90,
        "models": models,
        "weights": [base_weights[m] for m in models],
    }
    return _nan_to_none(out)
