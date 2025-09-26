# scripts/learn_weights_and_scenarios.py
# Fixes "linear future" by:
#  - Dropping 't_idx' (no time-index drift)
#  - Building SEASONAL, MEAN-REVERTING CPI scenarios (not linear steps)
#  - Adding mild nonlinear features (square & interaction)
#  - Keeping non-negative ensemble weights with gentle caps (avoid 0.92 single-model dominance)
#
# Inputs:
#   reports/backtest_series_forecast_{ridge,elasticnet,gb}.csv (≥2 of them)
#   data/cpi_clean.csv   (asof,cpi_yoy)
#   data/spx_actuals.csv (asof,actual_return)
#
# Outputs:
#   data/ensemble_weights.json
#   data/forecast_scenarios.json

import json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

REPORTS_DIR = "reports"
CPI_PATH = os.environ.get("SCL_CPI", "data/cpi_clean.csv")
ACTUALS_PATH = os.environ.get("SCL_ACTUALS", "data/spx_actuals.csv")
OUT_WEIGHTS = "data/ensemble_weights.json"
OUT_SCENARIOS = "data/forecast_scenarios.json"
os.makedirs("data", exist_ok=True)

# Horizons (months ahead)
H = 12

# Mean-reversion target & half-life for CPI YoY (decimals, e.g., 0.02 = 2%)
TARGET_YOY = float(os.environ.get("SCL_CPI_TARGET", "0.02"))
HALF_LIFE_M = int(os.environ.get("SCL_CPI_HALFLIFE_M", "12"))  # ~1 year to halve the gap
PHI = float(np.exp(np.log(0.5) / max(HALF_LIFE_M, 1)))         # AR(1) coefficient

# Ensemble guardrails (caps encourage diversity without hard-coding equal weights)
W_CAP = float(os.environ.get("SCL_W_CAP", "0.80"))    # max share any single base model may take
W_MIN_GB = float(os.environ.get("SCL_W_MIN_GB", "0.10"))  # ensure some nonlinearity by default

BASE_MODELS = ["ridge", "elasticnet", "gb"]

# ---------------- Backtest readers ----------------
def maybe_read_backtest_series(name: str) -> pd.DataFrame | None:
    path = os.path.join(REPORTS_DIR, f"backtest_series_forecast_{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["asof"])
    need = {"asof", "prediction", "actual_return"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path} must have columns {need}")
    return df.rename(columns={"prediction": name})[["asof", name, "actual_return"]]

def load_backtests() -> tuple[pd.DataFrame, list[str]]:
    dfs, present = [], []
    for m in BASE_MODELS:
        d = maybe_read_backtest_series(m)
        if d is not None:
            dfs.append(d); present.append(m)
    if len(dfs) < 2:
        raise SystemExit("Need at least 2 backtest series among ridge/elasticnet/gb. Re-run your backtest step.")
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on=["asof", "actual_return"], how="inner")
    df = df.sort_values("asof").reset_index(drop=True)
    return df, present

def learn_nonneg_weights(df_bt: pd.DataFrame, models_present: list[str]) -> Dict[str, float]:
    y = df_bt["actual_return"].values.astype(float)
    P = np.column_stack([df_bt[m].values for m in models_present]).astype(float)
    # OLS -> clamp to non-negative -> normalize
    w, *_ = np.linalg.lstsq(P, y, rcond=None)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 1e-12:
        w = np.ones(len(models_present)) / len(models_present)
    else:
        w = w / s
    # gentle cap to avoid single-model dominance
    w = np.minimum(w, W_CAP)
    w = w / max(w.sum(), 1e-12)
    # ensure some GB presence if available (adds curvature)
    if "gb" in models_present:
        idx = models_present.index("gb")
        if w[idx] < W_MIN_GB:
            delta = W_MIN_GB - w[idx]
            # take proportionally from the others
            idx_others = [i for i in range(len(w)) if i != idx and w[i] > 0]
            take = sum(w[i] for i in idx_others)
            if take > 0:
                for i in idx_others:
                    w[i] -= delta * (w[i] / take)
                w[idx] = W_MIN_GB
            # renormalize
            w = np.clip(w, 0.0, None)
            w = w / max(w.sum(), 1e-12)

    weights = {m: float(w[i]) for i, m in enumerate(models_present)}
    for m in BASE_MODELS:
        weights.setdefault(m, 0.0)
    return weights

# ---------------- Data + features ----------------
def load_merged() -> pd.DataFrame:
    cpi = pd.read_csv(CPI_PATH, parse_dates=["asof"]).sort_values("asof")
    act = pd.read_csv(ACTUALS_PATH, parse_dates=["asof"]).sort_values("asof")
    if "cpi_yoy" not in cpi.columns:
        raise SystemExit(f"{CPI_PATH} must include 'cpi_yoy'")
    if "actual_return" not in act.columns:
        raise SystemExit(f"{ACTUALS_PATH} must include 'actual_return'")
    df = cpi.merge(act[["asof","actual_return"]], on="asof", how="inner")
    df = df.rename(columns={"actual_return": "spx_return"}).reset_index(drop=True)
    return df

def make_timeaware_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("asof").reset_index(drop=True).copy()
    for l in [1, 2, 3, 6, 12]:
        df[f"cpi_yoy_l{l}"] = df["cpi_yoy"].shift(l)
    df["cpi_yoy_chg1"]   = df["cpi_yoy"].shift(1) - df["cpi_yoy"].shift(2)
    df["cpi_yoy_roll6"]  = df["cpi_yoy"].rolling(6).mean().shift(1)
    df["cpi_yoy_roll12"] = df["cpi_yoy"].rolling(12).mean().shift(1)
    df["cpi_yoy_vol6"]   = df["cpi_yoy"].rolling(6).std(ddof=0).shift(1)
    # mild nonlinearity to break linear slopes even if inputs drift smoothly
    df["cpi_yoy_l1_sq"]  = (df["cpi_yoy_l1"] ** 2)
    df["cpi_yoy_l1_x_chg1"] = df["cpi_yoy_l1"] * df["cpi_yoy_chg1"]
    return df

def feat_cols() -> List[str]:
    return [
        "cpi_yoy_l1","cpi_yoy_l2","cpi_yoy_l3","cpi_yoy_l6","cpi_yoy_l12",
        "cpi_yoy_chg1","cpi_yoy_roll6","cpi_yoy_roll12","cpi_yoy_vol6",
        "cpi_yoy_l1_sq","cpi_yoy_l1_x_chg1",
    ]

def build_targets_for_h(df_feat: pd.DataFrame, h: int) -> pd.DataFrame:
    df = df_feat.copy()
    df[f"target_h{h}"] = df["spx_return"].shift(-h)
    df = df.dropna(subset=feat_cols() + [f"target_h{h}"]).reset_index(drop=True)
    return df

# ---------------- Seasonal, mean-reverting CPI scenarios ----------------
def _seasonal_deltas(cpi_hist: pd.DataFrame) -> Dict[int, float]:
    """Estimate average monthly YoY change per month-of-year (deterministic, last ~10 years)."""
    df = cpi_hist.sort_values("asof").reset_index(drop=True).copy()
    df["delta"] = df["cpi_yoy"].diff()
    # use last 120 months if available
    tail = df.tail(min(120, len(df)))
    tail["month"] = tail["asof"].dt.month
    s = tail.groupby("month")["delta"].mean().to_dict()
    # default 0 for missing months
    for m in range(1, 13):
        s.setdefault(m, 0.0)
    return s

def extend_cpi_scenario(history: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Build future CPI_yoy path with:
      - AR(1) mean reversion to TARGET_YOY (half-life HALF_LIFE_M)
      - Month-of-year seasonal deltas from history
      - Scenario tilt on the target: ±0.30pp (0.003) around TARGET_YOY
    """
    tilt = {"baseline": 0.000, "optimistic": -0.003, "pessimistic": 0.003}.get(scenario, 0.0)
    target = TARGET_YOY + tilt

    last_date = history["asof"].iloc[-1]
    cur = float(history["cpi_yoy"].iloc[-1])
    seas = _seasonal_deltas(history)

    rows = []
    for step in range(1, H + 1):
        # next calendar month
        y = last_date.year + (last_date.month + step - 1) // 12
        mm = (last_date.month + step - 1) % 12 + 1
        d = pd.Timestamp(year=y, month=mm, day=1) + pd.offsets.MonthEnd(0)
        # seasonal component for that month
        s_delta = float(seas.get(mm, 0.0))
        # AR(1) mean reversion with seasonal increment
        nxt = target + PHI * (cur - target) + s_delta
        # clip to a sane range
        nxt = float(np.clip(nxt, -0.05, 0.10))  # [-5%, +10%] in decimals
        rows.append({"asof": d, "cpi_yoy": nxt})
        cur = nxt

    fut = pd.DataFrame(rows)
    return pd.concat([history[["asof","cpi_yoy"]], fut], ignore_index=True)

def features_for_future_h(cpi_path: pd.DataFrame, h: int) -> Tuple[np.ndarray, str]:
    df = cpi_path.copy().sort_values("asof").reset_index(drop=True)
    for l in [1,2,3,6,12]:
        df[f"cpi_yoy_l{l}"] = df["cpi_yoy"].shift(l)
    df["cpi_yoy_chg1"]   = df["cpi_yoy"].shift(1) - df["cpi_yoy"].shift(2)
    df["cpi_yoy_roll6"]  = df["cpi_yoy"].rolling(6).mean().shift(1)
    df["cpi_yoy_roll12"] = df["cpi_yoy"].rolling(12).mean().shift(1)
    df["cpi_yoy_vol6"]   = df["cpi_yoy"].rolling(6).std(ddof=0).shift(1)
    df["cpi_yoy_l1_sq"]  = (df["cpi_yoy"].shift(1) ** 2)
    df["cpi_yoy_l1_x_chg1"] = df["cpi_yoy"].shift(1) * df["cpi_yoy_chg1"]
    idx = len(df) - (H - h + 1)
    row = df.iloc[idx]
    X = row[feat_cols()].values.astype(float).reshape(1, -1)
    return X, row["asof"].strftime("%Y-%m-%d")

# ---------------- Main ----------------
def main():
    # (A) Learn ensemble weights from OOS backtests with gentle diversity caps
    df_bt, models_present = load_backtests()
    weights = learn_nonneg_weights(df_bt, models_present)
    with open(OUT_WEIGHTS, "w") as f:
        json.dump({"weights": weights, "n_samples": int(len(df_bt)), "models": models_present}, f, indent=2)
    print(f"[ok] wrote {OUT_WEIGHTS}: {weights}")

    # (B) Train direct multi-horizon models (no 't_idx'; slight nonlinear features)
    merged = load_merged()  # asof,cpi_yoy,spx_return
    feat = make_timeaware_features(merged).dropna(subset=feat_cols() + ["spx_return"]).reset_index(drop=True)
    Xcols = feat_cols()
    models, qmodels = {}, {}
    for h in range(1, H + 1):
        d = build_targets_for_h(feat, h)
        X = d[Xcols].values
        y_h = d[f"target_h{h}"].values
        models[h] = {
            "ridge": Ridge(alpha=5.0, random_state=0).fit(X, y_h),
            "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.3, random_state=0).fit(X, y_h),
            "gb": GradientBoostingRegressor(random_state=0).fit(X, y_h),
        }
        qmodels[h] = {
            "p10": GradientBoostingRegressor(loss="quantile", alpha=0.10, random_state=0).fit(X, y_h),
            "p90": GradientBoostingRegressor(loss="quantile", alpha=0.90, random_state=0).fit(X, y_h),
        }

    # (C) Forecast under SEASONAL mean-reverting CPI scenarios
    cpi_hist = merged[["asof","cpi_yoy"]]
    scenarios = ["baseline", "optimistic", "pessimistic"]
    out = {"weights": weights, "horizons": list(range(1, H + 1)), "scenarios": {}}

    # future date labels from baseline
    base_path = extend_cpi_scenario(cpi_hist, "baseline")
    fut_dates = []
    for h in range(1, H + 1):
        _, dstr = features_for_future_h(base_path, h)
        fut_dates.append(dstr)
    out["future_dates"] = fut_dates

    for sc in scenarios:
        path = extend_cpi_scenario(cpi_hist, sc)
        preds = {m: [] for m in BASE_MODELS}
        p10s, p90s = [], []
        for h in range(1, H + 1):
            Xf, _ = features_for_future_h(path, h)
            for m in BASE_MODELS:
                preds[m].append(float(models[h][m].predict(Xf)[0]))
            p10s.append(float(qmodels[h]["p10"].predict(Xf)[0]))
            p90s.append(float(qmodels[h]["p90"].predict(Xf)[0]))
        # ensemble with available weights (renormalize over positive entries)
        w_items = [(m, weights.get(m, 0.0)) for m in BASE_MODELS if weights.get(m, 0.0) > 0]
        if not w_items:
            w_items = [(m, 1.0) for m in BASE_MODELS]
        w_sum = sum(w for _, w in w_items)
        ens = []
        for i in range(H):
            val = sum((w / w_sum) * preds[m][i] for m, w in w_items)
            ens.append(float(val))

        out["scenarios"][sc] = {
            "ensemble": ens,
            "ridge": preds["ridge"],
            "elasticnet": preds["elasticnet"],
            "gb": preds["gb"],
            "p10": p10s,
            "p90": p90s,
        }

    with open(OUT_SCENARIOS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote {OUT_SCENARIOS}")

if __name__ == "__main__":
    main()
