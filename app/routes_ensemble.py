# app/routes_ensemble.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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

# ---------- small utils ----------
def _to_month_start(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

def _nan_to_none(x: Any) -> Any:
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if x is None: return None
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.ndarray, list, tuple)): return [_nan_to_none(v) for v in list(x)]
    if isinstance(x, dict): return {k: _nan_to_none(v) for k, v in x.items()}
    try:
        if pd.isna(x): return None
    except Exception:
        pass
    return x

def _read_csv(path: Path) -> pd.DataFrame:
    try: return pd.read_csv(path)
    except Exception: return pd.read_csv(path, sep=";")

def _detect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower: return lower[cand.lower()]
    return None

def _choose_pred_col(df: pd.DataFrame) -> Optional[str]:
    preferred = ["yhat","y_pred","prediction","pred","yhat_p50","median","mean","value","forecast","yhat_mean"]
    banned = {"actual","y","target","spx_ret","actual_return"}
    for name in preferred:
        c = _detect_col(list(df.columns), [name])
        if not c: continue
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().sum() >= 3 and float(ser.std(skipna=True) or 0.0) > 0.0:
            return c
    numeric = []
    for c in df.columns:
        if c in banned: continue
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().sum() >= 3:
            numeric.append((c, float(ser.std(skipna=True) or 0.0)))
    if not numeric: return None
    numeric.sort(key=lambda t: t[1], reverse=True)
    return numeric[0][0] if numeric[0][1] > 0.0 else None

# ---------- load/build tidy predictions ----------
def _build_predictions_from_reports() -> pd.DataFrame:
    files = sorted(REPORTS.glob("backtest_series_forecast_*.csv"))
    if not files:
        raise HTTPException(404, f"Missing {PRED_TIDY} and no reports/backtest_series_forecast_*.csv files found.")
    rows = []
    for fp in files:
        df = _read_csv(fp)
        if df is None or df.empty: continue
        date_col = _detect_col(list(df.columns), ["asof","month","date","ds","Date","timestamp","period","time"])
        if not date_col: continue
        pred_col = _choose_pred_col(df)
        if not pred_col: continue
        tmp = pd.DataFrame()
        tmp["asof"] = _to_month_start(df[date_col])
        tmp["yhat"] = pd.to_numeric(df[pred_col], errors="coerce")
        tmp["model"] = fp.stem.replace("backtest_series_forecast_", "")
        rows.append(tmp[["asof","model","yhat"]])
    if not rows:
        raise HTTPException(400, "No usable date/prediction columns in reports/*.csv.")
    tidy = pd.concat(rows, ignore_index=True).dropna(subset=["asof"]).sort_values(["asof","model"])
    try: tidy.to_csv(PRED_TIDY, index=False)
    except Exception: pass
    return tidy

def _load_predictions_tidy() -> pd.DataFrame:
    if PRED_TIDY.exists():
        df = _read_csv(PRED_TIDY)
        if df is not None and not df.empty:
            date_col = _detect_col(list(df.columns), ["asof","month","date","ds","Date","timestamp","period","time"])
            model_col = _detect_col(list(df.columns), ["model","name","estimator","model_name","algo","algorithm"])
            yhat_col = "yhat" if "yhat" in df.columns else _choose_pred_col(df)
            if date_col and model_col and yhat_col:
                tmp = pd.DataFrame()
                tmp["asof"] = _to_month_start(df[date_col])
                tmp["model"] = df[model_col].astype(str)
                tmp["yhat"] = pd.to_numeric(df[yhat_col], errors="coerce")
                tmp = tmp.dropna(subset=["asof"])
                if float(tmp["yhat"].std(skipna=True) or 0.0) > 0.0:
                    return tmp.sort_values(["asof","model"])
    return _build_predictions_from_reports()

def _load_actuals() -> pd.DataFrame:
    if not SPX_ACTUALS.exists():
        raise HTTPException(404, f"Missing {SPX_ACTUALS}")
    adf = _read_csv(SPX_ACTUALS)
    date_col = _detect_col(list(adf.columns), ["asof","month","date","ds","Date","timestamp","period","time"])
    if not date_col:
        raise HTTPException(400, "spx_actuals.csv needs a date column (asof/month/date/ds/Date/...).")
    if "actual" not in adf.columns:
        alt_val = _detect_col(list(adf.columns), ["actual_return","spx_ret","return","y","value"])
        if alt_val: adf = adf.rename(columns={alt_val: "actual"})
    if "actual" not in adf.columns:
        raise HTTPException(400, "spx_actuals.csv must have 'actual' (or one of: actual_return/spx_ret/return/y/value).")
    adf["asof"] = _to_month_start(adf[date_col])
    adf["actual"] = pd.to_numeric(adf["actual"], errors="coerce")
    adf = adf[["asof","actual"]].dropna(subset=["asof"]).drop_duplicates("asof")
    return adf.sort_values("asof")

# ---------- weights ----------
def _normalize_weights(models: List[str], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not models: return {}
    if not weights: return {m: 1.0/len(models) for m in models}
    w = {m: max(0.0, float(weights.get(m, 0.0))) for m in models}
    s = sum(w.values())
    return {m: (w[m]/s if s>0 else 1.0/len(models)) for m in models}

def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    # Euclidean projection onto simplex {w >= 0, sum w = 1}
    v = np.maximum(v, 0.0)
    if v.sum() <= 0: return np.ones_like(v)/len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w/s if s>0 else np.ones_like(v)/len(v)

def _auto_weights(wide: pd.DataFrame, acts: pd.Series, window: int = 60, l2: float = 1e-4) -> Dict[str,float]:
    # learn w by ridge on last 'window' months (drop rows with any NaN)
    cols = [c for c in wide.columns if c not in ("ensemble","model_std","actual")]
    joined = wide[cols].join(acts.rename("actual"), how="inner")
    joined = joined.dropna()
    if joined.empty or joined.shape[0] < max(12, len(cols)+2):
        # not enough data → equal weights
        return {c: 1.0/len(cols) for c in cols}
    tail = joined.tail(window)
    X = tail[cols].to_numpy(float)
    y = tail["actual"].to_numpy(float)
    # ridge (unconstrained), then project to simplex
    XtX = X.T @ X + l2 * np.eye(X.shape[1])
    Xty = X.T @ y
    try:
        w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    w = _project_to_simplex(w)
    return {c: float(w[i]) for i, c in enumerate(cols)}

# ---------- forecasting helpers ----------
def _fit_ar1(y: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if y.size < 8: return (0.0, 1.0, 0.0)  # carry-forward
    y_t, y_lag = y[1:], y[:-1]
    X0, X1 = np.ones_like(y_lag), y_lag
    XTX = np.array([[np.dot(X0, X0), np.dot(X0, X1)],[np.dot(X1, X0), np.dot(X1, X1)]])
    XTy = np.array([np.dot(X0, y_t), np.dot(X1, y_t)])
    try: c, phi = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError: c, phi = 0.0, 1.0
    resid = y_t - (c + phi * y_lag)
    sigma = float(np.std(resid)) if resid.size > 1 else 0.0
    return float(c), float(phi), sigma

def _ar1_path(y_last: float, steps: int, c: float, phi: float) -> List[float]:
    prev = y_last; out = []
    for _ in range(max(1, steps)):
        nxt = c + phi * prev
        out.append(float(nxt)); prev = nxt
    return out

def _rw_drift_params(y: np.ndarray, lookback: int = 12) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if y.size < 3: return (0.0, 0.0)
    diffs = np.diff(y[-max(2, lookback):])
    return float(np.nanmean(diffs) if diffs.size else 0.0), float(np.nanstd(diffs) if diffs.size>1 else 0.0)

def _rw_drift_path(y_last: float, steps: int, drift: float) -> List[float]:
    prev = y_last; out=[]
    for _ in range(max(1, steps)):
        nxt = prev + drift
        out.append(float(nxt)); prev = nxt
    return out

# ---------- conformal intervals ----------
def _conformal_residual_quantiles(ens: pd.Series, act: pd.Series, window: int = 36, alpha: float = 0.2) -> Tuple[float,float]:
    # residuals y - yhat (last 'window' with both available)
    df = pd.concat({"ens": ens, "act": act}, axis=1).dropna().tail(window)
    if df.empty or df.shape[0] < 12:
        return (-0.05, 0.05)  # sane fallback
    res = df["act"] - df["ens"]
    q_low = float(np.quantile(res, alpha/2))
    q_high = float(np.quantile(res, 1 - alpha/2))
    return (q_low, q_high)

# ---------- schemas ----------
class PastRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    auto_weights: bool = False

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
    auto_weights: bool = False

class FutureResponse(BaseModel):
    asof: str
    future_dates: List[str]
    future_prediction: List[Optional[float]]
    p10: List[Optional[float]]
    p90: List[Optional[float]]
    models: List[str]
    weights: List[float]

# ---------- endpoints ----------
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
    if pred.empty: raise HTTPException(404, "No predictions in requested range")

    models = sorted(pred["model"].dropna().astype(str).unique().tolist())
    wide = pred.pivot_table(index="asof", columns="model", values="yhat", aggfunc="mean").sort_index()
    acts_s = acts.set_index("asof")["actual"]

    # weights (manual or auto-learnt on last 60m)
    if req.auto_weights:
        base_w = _auto_weights(wide, acts_s, window=60, l2=1e-4)
    else:
        base_w = _normalize_weights(models, req.weights)

    # ensemble + cross-model std
    ens_vals, std_vals = [], []
    for _, row in wide.iterrows():
        avail = [m for m in models if pd.notna(row.get(m))]
        if not avail:
            ens_vals.append(np.nan); std_vals.append(np.nan); continue
        sub_w = _normalize_weights(avail, {m: base_w.get(m, 0.0) for m in avail})
        preds = np.array([float(row[m]) for m in avail], dtype=float)
        wv = np.array([sub_w[m] for m in avail], dtype=float)
        ens_vals.append(float(preds @ wv))
        std_vals.append(float(np.std(preds)) if len(preds) > 1 else 0.0)
    wide = wide.assign(ensemble=ens_vals, model_std=std_vals).join(acts_s, how="left")

    dates = [d.strftime("%Y-%m-%d") for d in wide.index]
    prediction = [None if pd.isna(v) else float(v) for v in wide["ensemble"].tolist()]
    actual = [None if "actual" not in wide or pd.isna(v) else float(v) for v in wide.get("actual", pd.Series(index=wide.index, dtype=float)).tolist()]

    # conformal band on past using residual quantiles
    ql, qh = _conformal_residual_quantiles(wide["ensemble"], wide["actual"], window=36, alpha=0.2)
    p10 = [None if y is None else float(y + ql) for y in prediction]
    p90 = [None if y is None else float(y + qh) for y in prediction]

    return _nan_to_none({
        "asof": datetime.now(timezone.utc).isoformat(),
        "dates": dates,
        "prediction": prediction,
        "actual": actual,
        "p10": p10,
        "p90": p90,
        "models": models,
        "weights": [base_w.get(m, 0.0) for m in models],
    })

@router.post("/future", response_model=FutureResponse)
def future(req: FutureRequest):
    pred = _load_predictions_tidy()
    acts = _load_actuals()

    models = sorted(pred["model"].dropna().astype(str).unique().tolist())
    wide = pred.pivot_table(index="asof", columns="model", values="yhat", aggfunc="mean").sort_index()
    acts_s = acts.set_index("asof")["actual"]

    # weights (manual or auto)
    if req.auto_weights:
        base_w = _auto_weights(wide, acts_s, window=60, l2=1e-4)
    else:
        base_w = _normalize_weights(models, req.weights)

    # historical ensemble
    ens_hist = []
    for _, row in wide.iterrows():
        avail = [m for m in models if pd.notna(row.get(m))]
        if not avail: ens_hist.append(np.nan); continue
        sub_w = _normalize_weights(avail, {m: base_w.get(m, 0.0) for m in avail})
        preds = np.array([float(row[m]) for m in avail], dtype=float)
        wv = np.array([sub_w[m] for m in avail], dtype=float)
        ens_hist.append(float(preds @ wv))
    ens_hist = np.asarray(ens_hist, dtype=float)
    ens_hist = ens_hist[np.isfinite(ens_hist)]
    if ens_hist.size == 0: raise HTTPException(404, "No ensemble history could be computed")
    tail = ens_hist[-60:] if ens_hist.size > 60 else ens_hist
    y_last = float(tail[-1])

    # AR(1) → drift fallback
    c, phi, sigma_ar = _fit_ar1(tail)
    use_drift = abs(phi) < 0.2
    if use_drift:
        drift, sigma_diff = _rw_drift_params(tail, lookback=12)
        path = _rw_drift_path(y_last, max(1, req.horizon), drift)
        sigma_base = sigma_diff
    else:
        path = _ar1_path(y_last, max(1, req.horizon), c, phi)
        sigma_base = sigma_ar

    # future dates start after last actual
    last_actual = acts["asof"].max()
    if pd.isna(last_actual): raise HTTPException(404, "No actuals found")
    idx = pd.date_range((last_actual + pd.offsets.MonthBegin(1)).normalize(), periods=max(1, req.horizon), freq="MS")
    future_dates = [d.strftime("%Y-%m-%d") for d in idx]

    # conformal band from recent residuals + √h widening
    # residual quantiles from the last 36m of (actual - ensemble)
    ens_series = pd.Series(ens_hist, index=wide.index[-len(ens_hist):])
    ql, qh = _conformal_residual_quantiles(ens_series, acts_s, window=36, alpha=0.2)
    disagree = float(np.nanstd(wide.iloc[-1].dropna().to_numpy(float))) if not wide.empty else 0.0
    base = float(disagree + max(abs(ql), abs(qh)) + sigma_base)
    bands = [float(base * math.sqrt(i + 1)) for i in range(len(path))]
    p10 = [float(v - b) for v, b in zip(path, bands)]
    p90 = [float(v + b) for v, b in zip(path, bands)]

    return _nan_to_none({
        "asof": datetime.now(timezone.utc).isoformat(),
        "future_dates": future_dates,
        "future_prediction": [float(v) for v in path],
        "p10": p10,
        "p90": p90,
        "models": models,
        "weights": [base_w.get(m, 0.0) for m in models],
    })
