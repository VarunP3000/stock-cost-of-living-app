# scripts/guardrails_bias_dependence.py
# Purpose:
# - Detect "always-up" optimism (too many positive predictions)
# - Verify CPI dependence (feature importance dominated by CPI features)
# - Compare against an "always-up" baseline on a recent OOS window
#
# Inputs (expected to exist; see step 2 if not):
#   data/cpi_clean.csv       with columns: asof,cpi_yoy
#   data/spx_actuals.csv     with columns: asof,actual_return
# Optional:
#   data/ensemble_weights.json  -> {"weights": {"ridge":..., "elasticnet":..., "gb":...}}
#
# Output:
#   data/guardrails_report.json  (metrics + pass/warn flags)

import json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

CPI_PATH = os.environ.get("SCL_CPI", "data/cpi_clean.csv")
ACT_PATH = os.environ.get("SCL_ACTUALS", "data/spx_actuals.csv")
WEIGHTS_PATH = "data/ensemble_weights.json"
OUT_PATH = "data/guardrails_report.json"
os.makedirs("data", exist_ok=True)

# --- Guard thresholds ---
TEST_HORIZON = 36         # last N months for OOS check
POS_RATE_MAX = 0.80       # if >80% preds are positive -> warn for "always-up" optimism
CPI_SHARE_MIN = 0.55      # CPI features should account for >=55% of importance/weight

def load_data() -> pd.DataFrame:
    if not os.path.exists(CPI_PATH):
        raise SystemExit(f"Missing {CPI_PATH}. Run your CPI cleaning step first.")
    if not os.path.exists(ACT_PATH):
        raise SystemExit(f"Missing {ACT_PATH}. You need actual SPX returns.")
    cpi = pd.read_csv(CPI_PATH, parse_dates=["asof"]).sort_values("asof")
    act = pd.read_csv(ACT_PATH, parse_dates=["asof"]).sort_values("asof")
    if "cpi_yoy" not in cpi.columns:
        raise SystemExit(f"{CPI_PATH} must include column 'cpi_yoy'")
    if "actual_return" not in act.columns:
        raise SystemExit(f"{ACT_PATH} must include column 'actual_return'")
    df = cpi.merge(act[["asof","actual_return"]], on="asof", how="inner").reset_index(drop=True)
    df = df.rename(columns={"actual_return": "spx_return"})
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("asof").reset_index(drop=True).copy()
    for l in [1,2,3,6,12]:
        df[f"cpi_yoy_l{l}"] = df["cpi_yoy"].shift(l)
    df["cpi_yoy_chg1"]   = df["cpi_yoy"].shift(1) - df["cpi_yoy"].shift(2)
    df["cpi_yoy_roll6"]  = df["cpi_yoy"].rolling(6).mean().shift(1)
    df["cpi_yoy_roll12"] = df["cpi_yoy"].rolling(12).mean().shift(1)
    df["cpi_yoy_vol6"]   = df["cpi_yoy"].rolling(6).std(ddof=0).shift(1)
    df["t_idx"]          = np.arange(len(df)).astype(float)  # known at t-1
    return df

def feat_cols() -> List[str]:
    return [
        "cpi_yoy_l1","cpi_yoy_l2","cpi_yoy_l3","cpi_yoy_l6","cpi_yoy_l12",
        "cpi_yoy_chg1","cpi_yoy_roll6","cpi_yoy_roll12","cpi_yoy_vol6",
        "t_idx",
    ]

def load_weights() -> Dict[str, float]:
    if os.path.exists(WEIGHTS_PATH):
        try:
            w = json.load(open(WEIGHTS_PATH, "r")).get("weights", {})
            if w: return {k: float(v) for k, v in w.items()}
        except Exception:
            pass
    return {"ridge": 1/3, "elasticnet": 1/3, "gb": 1/3}

def oos_window(df_feat: pd.DataFrame, cols: List[str], target: str, n_last: int):
    df_feat = df_feat.dropna(subset=cols + [target]).reset_index(drop=True)
    if len(df_feat) < n_last + 60:
        n_last = max(12, min(n_last, len(df_feat) // 3))
    train = df_feat.iloc[:-n_last]
    test  = df_feat.iloc[-n_last:]
    X_tr, y_tr = train[cols].values, train[target].values
    X_te, y_te = test[cols].values,  test[target].values
    return train, test, X_tr, y_tr, X_te, y_te

def model_fit_predict(X_tr, y_tr, X_te):
    ridge = Ridge(alpha=5.0, random_state=0).fit(X_tr, y_tr)
    enet  = ElasticNet(alpha=0.001, l1_ratio=0.3, random_state=0).fit(X_tr, y_tr)
    gb    = GradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
    return {
        "ridge": ridge, "elasticnet": enet, "gb": gb,
        "preds": {
            "ridge": ridge.predict(X_te),
            "elasticnet": enet.predict(X_te),
            "gb": gb.predict(X_te),
        }
    }

def metric_dir_acc(y_true, y_pred) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def cpi_importance_share(models: Dict[str, object], X_te, y_te, cols: List[str]) -> Dict[str, float]:
    """Return share of importance attributable to CPI features vs time index."""
    idx_time = cols.index("t_idx")
    is_cpi = np.array([c != "t_idx" for c in cols])

    shares = {}

    # Linear models: use |coef| magnitude
    for m in ["ridge","elasticnet"]:
        est = models[m]
        if hasattr(est, "coef_"):
            w = np.abs(est.coef_).astype(float)
            cpi_share = float(w[is_cpi].sum() / (w.sum() + 1e-12))
            shares[m] = cpi_share

    # GB: permutation importance (model-agnostic)
    est = models["gb"]
    try:
        pi = permutation_importance(est, X_te, y_te, n_repeats=10, random_state=0)
        imp = np.maximum(pi.importances_mean, 0.0)  # negative means noise
        cpi_share = float(imp[is_cpi].sum() / (imp.sum() + 1e-12))
        shares["gb"] = cpi_share
    except Exception:
        shares["gb"] = np.nan

    return shares

def main():
    df = load_data()
    df = make_features(df)
    cols = feat_cols()

    train, test, X_tr, y_tr, X_te, y_te = oos_window(df, cols, "spx_return", TEST_HORIZON)
    res = model_fit_predict(X_tr, y_tr, X_te)
    preds = res["preds"]
    weights = load_weights()

    # Ensemble predictions
    w_r = weights.get("ridge", 1/3); w_e = weights.get("elasticnet", 1/3); w_g = weights.get("gb", 1/3)
    w_sum = max(w_r + w_e + w_g, 1e-12)
    ens = (w_r*preds["ridge"] + w_e*preds["elasticnet"] + w_g*preds["gb"]) / w_sum

    # Metrics
    metrics = {}
    always_up_acc = metric_dir_acc(y_te, np.ones_like(y_te))  # baseline: always positive

    for name, yhat in list(preds.items()) + [("ensemble", ens)]:
        rmse = float(np.sqrt(mean_squared_error(y_te, yhat)))
        mae  = float(mean_absolute_error(y_te, yhat))
        dir_acc = metric_dir_acc(y_te, yhat)
        pos_rate = float(np.mean(yhat > 0))
        metrics[name] = {
            "rmse": rmse, "mae": mae, "directional_accuracy": dir_acc, "positive_rate": pos_rate
        }

    # Importance shares
    shares = cpi_importance_share(
        {"ridge": res["ridge"], "elasticnet": res["elasticnet"], "gb": res["gb"]},
        X_te, y_te, cols
    )

    # Guard flags
    guards = {}
    for name, m in metrics.items():
        guards[f"{name}_optimism_warn"] = bool(m["positive_rate"] > POS_RATE_MAX and m["directional_accuracy"] <= always_up_acc)
    for name in ["ridge","elasticnet","gb"]:
        if name in shares and not np.isnan(shares[name]):
            guards[f]()
