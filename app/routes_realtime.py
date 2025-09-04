# app/routes_realtime.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
import traceback

router = APIRouter(prefix="/realtime", tags=["realtime"])

ARTIFACTS_DIR = Path("models/artifacts")
PANEL_PRIMARY = Path("data/processed/spx_cpi_global_clean.csv")
PANEL_FALLBACK = Path("data/processed/spx_cpi_global.csv")
ACTUALS_PATH = Path("data/spx_actuals.csv")  # supports actual/actual_return/spx_ret etc.

# -------------------- Artifact loading --------------------
def _discover_models() -> Dict[str, Any]:
    if not ARTIFACTS_DIR.exists():
        raise HTTPException(500, f"Missing artifacts dir: {ARTIFACTS_DIR}")

    models: Dict[str, Any] = {}
    for p in ARTIFACTS_DIR.glob("*.pkl"):
        try:
            m = joblib.load(p)
            key = p.stem.lower()
            models[key] = m
        except Exception as e:
            warnings.warn(f"Failed to load {p}: {e}")
    if not models:
        raise HTTPException(500, f"No .pkl models found in {ARTIFACTS_DIR}")
    return models

MODELS_CACHE: Dict[str, Any] = {}
def _models() -> Dict[str, Any]:
    global MODELS_CACHE
    if not MODELS_CACHE:
        MODELS_CACHE = _discover_models()
    return MODELS_CACHE

def _is_predictor(m: Any) -> bool:
    """Return True if the artifact can produce predictions."""
    if hasattr(m, "predict") or hasattr(m, "predict_proba"):
        return True
    if hasattr(m, "inplace_predict"):
        return True
    if hasattr(m, "predict") and "lightgbm" in m.__class__.__module__:
        return True
    return False

def _predictor_subset(ms: Dict[str, Any]) -> Dict[str, Any]:
    return {name: m for name, m in ms.items() if _is_predictor(m)}

# -------------------- Data helpers --------------------
def _to_month_start(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

def _load_panel() -> pd.DataFrame:
    if PANEL_PRIMARY.exists():
        df = pd.read_csv(PANEL_PRIMARY)
    elif PANEL_FALLBACK.exists():
        df = pd.read_csv(PANEL_FALLBACK)
    else:
        raise HTTPException(500, f"Missing processed panel: {PANEL_PRIMARY} or {PANEL_FALLBACK}")
    date_col = next((c for c in ["month","asof","date","ds","Date"] if c in df.columns), None)
    if not date_col:
        raise HTTPException(500, "Panel needs a date-like column (month/asof/date/ds/Date)")
    df["month"] = _to_month_start(df[date_col])
    return df

def _load_actuals() -> pd.DataFrame:
    if not ACTUALS_PATH.exists():
        raise HTTPException(500, f"Missing {ACTUALS_PATH}")
    adf = pd.read_csv(ACTUALS_PATH)
    date_col = next((c for c in ["asof","month","date","ds","Date"] if c in adf.columns), None)
    if not date_col:
        raise HTTPException(500, "spx_actuals.csv needs a date column (asof/month/date/ds/Date)")
    if "actual" not in adf.columns:
        for c in ["actual_return","spx_ret","return","y","value"]:
            if c in adf.columns:
                adf = adf.rename(columns={c:"actual"})
                break
    if "actual" not in adf.columns:
        raise HTTPException(500, "spx_actuals.csv must have 'actual' (or one of actual_return/spx_ret/return/y/value)")
    adf["asof"] = _to_month_start(adf[date_col])
    adf["actual"] = pd.to_numeric(adf["actual"], errors="coerce")
    adf = adf[["asof","actual"]].dropna(subset=["asof"]).drop_duplicates("asof").sort_values("asof")
    return adf

# -------------------- Feature building --------------------
def _ensure_2d_row(df_or_dict: Any, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert features -> DataFrame with one row.
    If 'columns' provided (model's feature_names_in_), reindex to that exact schema
    and fill *all* missing values (any dtype) with 0.0 so sklearn never sees NaN.
    """
    if isinstance(df_or_dict, dict):
        df = pd.DataFrame([df_or_dict])
    elif isinstance(df_or_dict, pd.DataFrame):
        df = df_or_dict.copy()
    else:
        raise HTTPException(400, "features must be an object (dict)")

    # soft numeric conversion
    for c in df.columns:
        if df[c].dtype == "object":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conv = pd.to_numeric(df[c], errors="ignore")
                df[c] = conv

    if columns:
        base = pd.DataFrame(columns=columns)
        df = pd.concat([base, df], ignore_index=True).tail(1)
        df = df.fillna(0.0)  # IMPORTANT: fill all NaNs
        df = df[columns]
        return df, columns

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0.0)
    return df, list(df.columns)

def _panel_features_for_date(panel: pd.DataFrame, ts: pd.Timestamp) -> Dict[str, Any]:
    """
    Minimal demo feature builder. If a model requests columns we don't make,
    the adapter will align and fill them with 0.0 (safe default).
    """
    sub = panel[panel["month"] <= ts]
    if sub.empty:
        raise HTTPException(404, f"No panel rows <= {ts.date()}")
    row = sub.tail(1).iloc[0]
    cand = [c for c in ["cpi_yoy","cpi_mom","spx_ret_1m","spx_close","cpi_index"] if c in sub.columns]
    feat = {f"feat_{c}": (float(row[c]) if pd.notna(row[c]) else 0.0) for c in cand}
    feat["bias"] = 1.0
    return feat

# -------------------- Prediction + metrics --------------------
def _predict_one(model: Any, X: pd.DataFrame, model_name: str) -> float:
    """
    Returns a scalar prediction.
    - If classifier with predict_proba: uses positive-class probability.
    - Else uses predict; if array -> first element.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return float(proba[0, 1])
            return float(proba.reshape(-1)[0])

        if hasattr(model, "predict"):
            y = np.asarray(model.predict(X))
            return float(y.reshape(-1)[0])

        if hasattr(model, "inplace_predict"):
            y = np.asarray(model.inplace_predict(X))
            return float(y.reshape(-1)[0])

        if hasattr(model, "predict") and "lightgbm" in model.__class__.__module__:
            y = np.asarray(model.predict(X))
            return float(y.reshape(-1)[0])

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' prediction failed: {e}\nColumns: {list(X.columns)}\n{tb}",
        )
    raise HTTPException(400, f"Unsupported model type for '{model_name}': {type(model)}")

def _complete_weights(model_names: List[str], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not model_names:
        raise HTTPException(500, "No models available")
    if not weights:
        return {m: 1.0 / len(model_names) for m in model_names}
    w = {m: float(weights.get(m, 0.0)) for m in model_names}
    s = sum(w.values())
    if s <= 0:
        return {m: 1.0 / len(model_names) for m in model_names}
    if s != 1.0:
        w = {m: v / s for m, v in w.items()}
    return w

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "hit_rate": float("nan")}
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs((yp - yt) / yt)))
    if len(yt) < 2 or len(yp) < 2:
        hit = float("nan")
    else:
        s_true = np.sign(yt[1:] - yt[:-1])
        s_pred = np.sign(yp[1:] - yp[:-1])
        hit = float((s_true == s_pred).mean())
    return {"rmse": rmse, "mae": mae, "mape": mape, "hit_rate": hit}

# -------------------- Schemas --------------------
class RealtimePredictRequest(BaseModel):
    features: Dict[str, Any]
    weights: Optional[Dict[str, float]] = None

class RealtimePredictResponse(BaseModel):
    models: List[str]
    per_model: Dict[str, float]
    ensemble: float
    used_weights: Dict[str, float]
    features_used: List[str]

class RealtimeFromDateRequest(BaseModel):
    date: str
    weights: Optional[Dict[str, float]] = None

class RealtimeFromDateResponse(BaseModel):
    date: str
    features: Dict[str, Any]
    models: List[str]
    per_model: Dict[str, float]
    ensemble: float
    used_weights: Dict[str, float]

class FeatureSpecsResponse(BaseModel):
    models: List[str]
    feature_specs: Dict[str, List[str]]

class PastPanelRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    weights: Optional[Dict[str, float]] = None

class PastPanelResponse(BaseModel):
    points: List[Dict[str, Any]]
    metrics: Dict[str, float]
    model_weights: Dict[str, float]
    model_list: List[str]
    model_panel: List[Dict[str, Any]]

class FuturePanelRequest(BaseModel):
    horizon: int = 6
    weights: Optional[Dict[str, float]] = None

class FuturePanelResponse(BaseModel):
    forecast: List[Dict[str, Any]]
    confidence_band: List[Dict[str, Any]]
    model_weights: Dict[str, float]
    model_list: List[str]

# -------------------- Endpoints --------------------
@router.get("/models")
def realtime_models():
    all_models = _models()
    preds = _predictor_subset(all_models)
    return {"models": sorted(preds.keys()), "artifact_dir": str(ARTIFACTS_DIR.resolve())}

@router.get("/feature_specs", response_model=FeatureSpecsResponse)
def feature_specs():
    ms = _predictor_subset(_models())
    specs: Dict[str, List[str]] = {}
    for name, m in ms.items():
        cols = getattr(m, "feature_names_in_", None)
        if cols is None:
            try:
                cols = list(getattr(m, "feature_names_in_", []))
            except Exception:
                cols = []
        specs[name] = list(cols) if cols is not None else []
    return {"models": sorted(ms.keys()), "feature_specs": specs}

@router.post("/predict", response_model=RealtimePredictResponse)
def realtime_predict(req: RealtimePredictRequest):
    ms_all = _models()
    ms = _predictor_subset(ms_all)
    names = sorted(ms.keys())
    weights = _complete_weights(names, req.weights)

    per_model: Dict[str, float] = {}
    X_any, cols_any = _ensure_2d_row(req.features)
    for name in names:
        m = ms[name]
        cols = getattr(m, "feature_names_in_", None)
        if cols is None:
            try:
                cols = list(getattr(m, "feature_names_in_", []))
            except Exception:
                cols = None
        if cols is None or len(cols) == 0:
            X, _ = _ensure_2d_row(req.features, list(cols_any))
        else:
            X, _ = _ensure_2d_row(req.features, list(cols))
        per_model[name] = _predict_one(m, X, name)

    ensemble = float(sum(per_model[k] * weights[k] for k in names))
    return {
        "models": names,
        "per_model": per_model,
        "ensemble": ensemble,
        "used_weights": weights,
        "features_used": list(X_any.columns),
    }

@router.post("/predict_from_date", response_model=RealtimeFromDateResponse)
def realtime_predict_from_date(req: RealtimeFromDateRequest):
    panel = _load_panel()
    ts = pd.to_datetime(req.date).to_period("M").to_timestamp()
    feat = _panel_features_for_date(panel, ts)

    ms = _predictor_subset(_models())
    names = sorted(ms.keys())
    weights = _complete_weights(names, req.weights)

    per_model: Dict[str, float] = {}
    X_any, cols_any = _ensure_2d_row(feat)
    for name in names:
        m = ms[name]
        cols = getattr(m, "feature_names_in_", None)
        if cols is None:
            try:
                cols = list(getattr(m, "feature_names_in_", []))
            except Exception:
                cols = None
        if cols is None or len(cols) == 0:
            X, _ = _ensure_2d_row(feat, list(cols_any))
        else:
            X, _ = _ensure_2d_row(feat, list(cols))
        per_model[name] = _predict_one(m, X, name)

    ensemble = float(sum(per_model[k] * weights[k] for k in names))
    return {
        "date": req.date,
        "features": feat,
        "models": names,
        "per_model": per_model,
        "ensemble": ensemble,
        "used_weights": weights,
    }

@router.post("/past_panel", response_model=PastPanelResponse)
def past_panel(req: PastPanelRequest):
    panel = _load_panel()
    acts = _load_actuals()

    timeline = panel["month"].dropna().drop_duplicates().sort_values()
    if req.start:
        timeline = timeline[timeline >= pd.to_datetime(req.start)]
    if req.end:
        timeline = timeline[timeline <= pd.to_datetime(req.end)]
    if timeline.empty:
        raise HTTPException(404, "No panel months in requested range")

    ms = _predictor_subset(_models())
    names = sorted(ms.keys())
    weights = _complete_weights(names, req.weights)

    rows = []
    model_rows = []

    for ts in timeline:
        feat = _panel_features_for_date(panel, ts)
        X_any, cols_any = _ensure_2d_row(feat)

        preds = {}
        for name in names:
            m = ms[name]
            cols = getattr(m, "feature_names_in_", None)
            if cols is None:
                try:
                    cols = list(getattr(m, "feature_names_in_", []))
                except Exception:
                    cols = None
            if cols is None or len(cols) == 0:
                X, _ = _ensure_2d_row(feat, list(cols_any))
            else:
                X, _ = _ensure_2d_row(feat, list(cols))
            preds[name] = _predict_one(m, X, name)

        ens = float(sum(preds[k] * weights[k] for k in names))
        std = float(np.std([preds[k] for k in names])) if len(names) > 1 else 0.0

        rows.append({"asof": ts.strftime("%Y-%m-%d"), "ensemble": ens, "actual": None, "model_std": std})
        per_row = {"asof": ts.strftime("%Y-%m-%d")}
        per_row.update({k: preds[k] for k in names})
        model_rows.append(per_row)

    df_points = pd.DataFrame(rows)
    df_model = pd.DataFrame(model_rows)

    # ---- merge with actuals; coalesce actual_y / actual_x → actual
    acts_str = acts.assign(asof_str=acts["asof"].dt.strftime("%Y-%m-%d"))
    merged = df_points.merge(
        acts_str[["asof_str", "actual"]],
        left_on="asof",
        right_on="asof_str",
        how="left",
        suffixes=("_x", "_y"),
    )
    # prefer the incoming actual from acts (actual_y), else keep placeholder (actual_x), else None
    if "actual_y" in merged.columns:
        merged["actual"] = merged["actual_y"]
    elif "actual_x" in merged.columns:
        merged["actual"] = merged["actual_x"]
    # clean up helper columns
    merged = merged.drop(columns=[c for c in ["asof_str", "actual_x", "actual_y"] if c in merged.columns], errors="ignore")
    merged["actual"] = pd.to_numeric(merged.get("actual", pd.Series(dtype=float)), errors="coerce")

    metrics = _metrics(
        merged["actual"].to_numpy(dtype=float),
        merged["ensemble"].to_numpy(dtype=float),
    )

    return {
        "points": merged[["asof","ensemble","actual","model_std"]].to_dict(orient="records"),
        "metrics": metrics,
        "model_weights": weights,
        "model_list": names,
        "model_panel": df_model.to_dict(orient="records"),
    }

@router.post("/future_panel", response_model=FuturePanelResponse)
def future_panel(req: FuturePanelRequest):
    panel = _load_panel()
    acts = _load_actuals()

    ms = _predictor_subset(_models())
    names = sorted(ms.keys())
    weights = _complete_weights(names, req.weights)

    last_actual = acts["asof"].max()
    start_future = (last_actual + pd.offsets.MonthBegin(1)).normalize()
    dates = pd.date_range(start_future, periods=req.horizon, freq="MS")

    last_feat = _panel_features_for_date(panel, last_actual)
    X_any, cols_any = _ensure_2d_row(last_feat)

    per_model_pred_last = []
    for name in names:
        m = ms[name]
        cols = getattr(m, "feature_names_in_", None)
        if cols is None:
            try:
                cols = list(getattr(m, "feature_names_in_", []))
            except Exception:
                cols = None
        if cols is None or len(cols) == 0:
            X, _ = _ensure_2d_row(last_feat, list(cols_any))
        else:
            X, _ = _ensure_2d_row(last_feat, list(cols))
        per_model_pred_last.append(_predict_one(m, X, name))
    last_std = float(np.std(per_model_pred_last)) if len(per_model_pred_last) > 1 else 0.0

    # quick MAE over last 24 months
    try:
        all_months = panel["month"].dropna().drop_duplicates().sort_values()
        back_months = all_months[all_months <= last_actual].tail(24)
        ens_vals, act_vals = [], []
        for ts in back_months:
            feat = _panel_features_for_date(panel, ts)
            X_any2, cols_any2 = _ensure_2d_row(feat)
            preds = []
            for name in names:
                m = ms[name]
                cols = getattr(m, "feature_names_in_", None)
                if cols is None:
                    try:
                        cols = list(getattr(m, "feature_names_in_", []))
                    except Exception:
                        cols = None
                if cols is None or len(cols) == 0:
                    X2, _ = _ensure_2d_row(feat, list(cols_any2))
                else:
                    X2, _ = _ensure_2d_row(feat, list(cols))
                preds.append(_predict_one(m, X2, name))
            ens_vals.append(float(sum(preds[i] * _complete_weights(names, weights)[names[i]] for i in range(len(names)))))
            arow = acts[acts["asof"] == ts]
            act_vals.append(float(arow["actual"].iloc[0]) if not arow.empty else np.nan)
        m = _metrics(np.array(act_vals, dtype=float), np.array(ens_vals, dtype=float))
        mae_all = 0.0 if np.isnan(m["mae"]) else float(m["mae"])
    except Exception:
        mae_all = 0.0

    forecast = []
    confidence_band = []
    for dt in dates:
        yhat = float(sum(per_model_pred_last[i] * _complete_weights(names, weights)[names[i]] for i in range(len(names))))
        forecast.append({"asof": dt.strftime("%Y-%m-%d"), "yhat": yhat, "disagreement": last_std})
        band = last_std + mae_all
        confidence_band.append({"asof": dt.strftime("%Y-%m-%d"), "yhat_lo": yhat - band, "yhat_hi": yhat + band})

    return {
        "forecast": forecast,
        "confidence_band": confidence_band,
        "model_weights": _complete_weights(names, weights),
        "model_list": names,
    }
