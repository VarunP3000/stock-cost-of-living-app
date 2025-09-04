# scripts/backtest_enriched.py
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
PLOTS = REPORTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)


def load_predictions():
    candidates = [
        REPORTS / "backtest_joined.csv",        # if you already have a joined file
        ROOT / "data" / "predictions_tidy.csv", # logged predictions (no actuals)
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p, parse_dates=["asof"], low_memory=False)
                return df
            except Exception:
                pass
    raise SystemExit(
        "No predictions file found. Expected reports/backtest_joined.csv or data/predictions_tidy.csv."
    )


def ensure_actuals(df):
    pred_col = next(
        (c for c in df.columns if c in {"prediction", "pred", "yhat", "forecast"}), None
    )
    if pred_col is None:
        raise SystemExit(f"Couldn't find a prediction column in: {df.columns.tolist()}")
    if "actual_return" in df.columns:
        return df, pred_col, "actual_return"

    # Try to merge actuals from data/spx_actuals.csv
    actuals_path = ROOT / "data" / "spx_actuals.csv"
    if not actuals_path.exists():
        msg = (
            "Missing data/spx_actuals.csv (actual SPX returns). "
            "Create it first (example pipeline):\n"
            "  python scripts/backtest_from_log.py\n"
            "  python scripts/backtest_join_actuals.py data/spx_actuals.csv\n"
            "Or write data/spx_actuals.csv with columns: asof,actual_return"
        )
        raise SystemExit(msg)

    actuals = pd.read_csv(actuals_path, parse_dates=["asof"])
    need_cols = {"asof", "actual_return"}
    if not need_cols.issubset(actuals.columns):
        raise SystemExit(
            f"{actuals_path} must have columns {need_cols}. Got: {actuals.columns.tolist()}"
        )

    merged = pd.merge(df, actuals[["asof", "actual_return"]], on="asof", how="inner")
    if merged.empty:
        raise SystemExit(
            "After merging with spx_actuals.csv, no overlapping dates were found. Check 'asof' formats."
        )
    return merged, pred_col, "actual_return"


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def mae(e):
    return float(np.mean(np.abs(e)))


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def directional_acc(y, yhat):
    return float(
        np.mean(((yhat >= 0) & (y >= 0)) | ((yhat < 0) & (y < 0))))


def main():
    df = load_predictions()
    df, pred_col, actual_col = ensure_actuals(df)
    req = {"asof", "route", pred_col, actual_col}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(
            f"Backtest file missing required cols {missing}. Got: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[pred_col, actual_col, "route"]).sort_values("asof")

    rows = []
    for route, g in df.groupby("route"):
        y = g[actual_col].astype(float).values
        yhat = g[pred_col].astype(float).values
        e = yhat - y
        rows.append(
            {
                "route": route,
                "n": int(len(g)),
                "mae": mae(e),
                "rmse": rmse(e),
                "r2": r2(y, yhat),
                "directional_accuracy": directional_acc(y, yhat),
                "asof_min": str(pd.to_datetime(g["asof"].min()).date()),
                "asof_max": str(pd.to_datetime(g["asof"].max()).date()),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        )

        # Quick chart
        fig = plt.figure()
        plt.plot(g["asof"], g[actual_col], label="Actual")
        plt.plot(g["asof"], g[pred_col], label="Predicted")
        plt.title(f"{route}: Actual vs Predicted")
        plt.legend()
        fig.tight_layout()
        out = PLOTS / f"{route.replace('/','_')}_actual_vs_pred.png"
        fig.savefig(out, dpi=144)
        plt.close(fig)
        print(f"Saved {out}")

    summary = pd.DataFrame(rows).sort_values("route")
    out_csv = REPORTS / "backtest_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(summary)} rows")


if __name__ == "__main__":
    main()
