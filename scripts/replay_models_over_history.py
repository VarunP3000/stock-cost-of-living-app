#!/usr/bin/env python
# scripts/replay_models_over_history.py
"""
Replays monthly backtests over history and writes:
  reports/backtest_series_forecast_<model>.csv

Compatible CLI:
  --cpi PATH_TO_CPI_WIDE_CSV
  --actuals PATH_TO_SPX_ACTUALS_CSV (asof, actual_return)
  --models ridge elasticnet gb ensemble
  [--window MONTHS] rolling train window (default 120)
  [--lags 1] number of return lags to include (default 1)

Notes
- This version is robust to NaNs in CPI (imputes means) and skips months
  that have no usable features instead of crashing.
- It trains each model on data up-to-(t-1) and predicts month t (proper backtest).
- 'ensemble' is the simple average of available base models that month.
"""

import argparse
import pathlib
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

OUTDIR = pathlib.Path("reports")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _infer_date_col(df: pd.DataFrame) -> str:
    best = None
    good_best = -1
    for c in df.columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce")
            good = dt.notna().sum()
            if good > good_best and good > 3:
                best = c
                good_best = good
        except Exception:
            pass
    if best is None:
        raise ValueError("Could not infer a date/asof column.")
    return best


def _pca_factors(X: pd.DataFrame, n_comp: int = 3) -> Optional[np.ndarray]:
    """
    Mean-impute NaNs, drop all-NaN or constant columns, center, PCA.
    Returns None if not enough rows/cols.
    """
    if X is None or len(X) == 0:
        return None

    W = X.copy()
    # remove non-numeric columns quietly
    for c in list(W.columns):
        if not np.issubdtype(W[c].dtype, np.number):
            with np.errstate(all="ignore"):
                W[c] = pd.to_numeric(W[c], errors="coerce")

    # drop all-NaN columns
    W = W.dropna(axis=1, how="all")
    if W.shape[1] == 0:
        return None

    # drop ~constant columns
    var = W.var(skipna=True)
    W = W.loc[:, var > 0]
    if W.shape[1] == 0:
        return None

    # impute
    imp = SimpleImputer(strategy="mean")
    A = imp.fit_transform(W.values.astype(float))

    # need at least 1 row and col
    if A.shape[0] < 1 or A.shape[1] < 1:
        return None

    # center
    A = A - np.mean(A, axis=0, keepdims=True)

    n_comp_eff = int(min(n_comp, A.shape[0], A.shape[1]))
    if n_comp_eff < 1:
        return None

    pca = PCA(n_components=n_comp_eff)
    F = pca.fit_transform(A)
    return F


def _build_design(cpi_path: pathlib.Path, actuals_path: pathlib.Path, n_lags: int = 1):
    # CPI wide file → ensure monthly 'asof'
    cpi = pd.read_csv(cpi_path)
    date_c = _infer_date_col(cpi)
    cpi["asof"] = pd.to_datetime(cpi[date_c], errors="coerce").dt.to_period("M").dt.to_timestamp()
    cpi = cpi.dropna(subset=["asof"]).sort_values("asof")
    # drop the date col duplicate
    if date_c != "asof":
        cpi = cpi.drop(columns=[date_c], errors="ignore")

    # PCA factors on full CPI (stable transform across time)
    cpi_feats = cpi.drop(columns=["asof"], errors="ignore")
    F = _pca_factors(cpi_feats, n_comp=3)
    if F is None:
        raise ValueError("PCA found no usable CPI features (all NaN/constant?).")
    F = pd.DataFrame(F, columns=[f"pc{i+1}" for i in range(F.shape[1])])
    F["asof"] = cpi["asof"].values
    design = F.copy()

    # load actuals
    act = pd.read_csv(actuals_path, parse_dates=["asof"])
    act["asof"] = act["asof"].dt.to_period("M").dt.to_timestamp()
    act = act.dropna(subset=["asof", "actual_return"]).sort_values("asof")

    # join on asof
    df = pd.merge(design, act, on="asof", how="inner")

    # add lagged returns
    for L in range(1, n_lags + 1):
        df[f"ret_lag{L}"] = df["actual_return"].shift(L)

    # final cleaning
    df = df.dropna().reset_index(drop=True)
    if len(df) < 24:
        raise ValueError(
            f"Not enough overlapping rows after cleaning (got {len(df)}). "
            f"Ensure CPI and ACT ranges overlap and contain numeric data."
        )

    features = [c for c in df.columns if c.startswith("pc") or c.startswith("ret_lag")]
    X = df[features].copy()
    y = df["actual_return"].copy()
    dates = df["asof"].copy()
    return X, y, dates, features


def _fit_predict_series(
    model_name: str, X: pd.DataFrame, y: pd.Series, dates: pd.Series, window: int
) -> pd.DataFrame:
    """
    Rolling-origin backtest:
      for t from start..end:
        fit on [t-window, t-1], predict t.
    """
    if model_name == "ridge":
        mk = lambda: Ridge(alpha=1.0, random_state=0)
    elif model_name == "elasticnet":
        mk = lambda: ElasticNet(alpha=0.0005, l1_ratio=0.2, random_state=0, max_iter=10000)
    elif model_name == "gb":
        mk = lambda: GradientBoostingRegressor(random_state=0)
    else:
        raise ValueError(f"Unknown base model: {model_name}")

    rows = []
    n = len(X)
    start = max(window, 24)  # require some minimum history
    for t in range(start, n):
        tr_lo = max(0, t - window)
        tr_hi = t  # up to t-1 inclusive
        Xtr, ytr = X.iloc[tr_lo:tr_hi], y.iloc[tr_lo:tr_hi]
        if len(Xtr) < 8:
            continue
        mdl = mk()
        try:
            mdl.fit(Xtr, ytr)
            yhat = float(mdl.predict(X.iloc[[t]])[0])
        except Exception:
            # skip any degenerate month
            continue
        rows.append(
            {
                "asof": dates.iloc[t],
                "prediction": yhat,
                "actual_return": float(y.iloc[t]),
                "model": model_name,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ridge", "elasticnet", "gb", "ensemble"])
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--actuals", required=True)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--lags", type=int, default=1)
    args = ap.parse_args()

    cpi_path = pathlib.Path(args.cpi)
    act_path = pathlib.Path(args.actuals)
    if not cpi_path.exists():
        sys.exit(f"Missing CPI file: {cpi_path}")
    if not act_path.exists():
        sys.exit(f"Missing ACTUALS file: {act_path}")

    X, y, dates, features = _build_design(cpi_path, act_path, n_lags=args.lags)

    # Fit base models
    outputs = {}
    base_models: List[str] = [m for m in args.models if m != "ensemble"]
    for m in base_models:
        print(f"[replay] training backtest for: {m}")
        dfm = _fit_predict_series(m, X, y, dates, window=args.window)
        if not dfm.empty:
            outputs[m] = dfm
            outpath = OUTDIR / f"backtest_series_forecast_{m}.csv"
            dfm.to_csv(outpath, index=False)
            print(f"Wrote {outpath} with {len(dfm)} rows")
        else:
            print(f"[warn] no rows produced for model={m}")

    # Ensemble = mean of available base predictions at same date
    if "ensemble" in args.models:
        if outputs:
            # concatenate and pivot to combine same-date predictions
            all_base = pd.concat(outputs.values(), ignore_index=True)
            piv = all_base.pivot_table(
                index="asof",
                columns="model",
                values="prediction",
                aggfunc="mean"
            )
            piv["ensemble"] = piv.mean(axis=1, skipna=True)

            # bring back actuals
            act_df = all_base[["asof", "actual_return"]].drop_duplicates("asof").set_index("asof")
            ens = piv.join(act_df, how="inner").reset_index()

            out = ens[["asof", "ensemble", "actual_return"]].rename(
                columns={"ensemble": "prediction"}
            )
            out["model"] = "ensemble"
            out = out.sort_values("asof")
            outpath = OUTDIR / "backtest_series_forecast_ensemble.csv"
            out.to_csv(outpath, index=False)
            print(f"Wrote {outpath} with {len(out)} rows")
        else:
            print("[warn] ensemble skipped (no base model outputs)")

    print("Done.")


if __name__ == "__main__":
    main()
