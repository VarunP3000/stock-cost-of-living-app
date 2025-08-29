import sys
from pathlib import Path
import pandas as pd

PRED_TIDY = Path("data/predictions_tidy.csv")
ACTUALS = Path(sys.argv[1] if len(sys.argv) > 1 else "data/spx_actuals.csv")
OUT_SUMMARY = Path("reports/backtest_summary.csv")
OUT_ERRORS = Path("reports/backtest_errors.csv")

if not PRED_TIDY.exists():
    raise SystemExit(f"Missing {PRED_TIDY}. Run scripts/backtest_from_log.py first.")
if not ACTUALS.exists():
    raise SystemExit(f"Missing {ACTUALS}. Provide an actuals CSV: asof,actual_return")

# --- Load + parse dates (tz-safe) ---
pred = pd.read_csv(PRED_TIDY)
act = pd.read_csv(ACTUALS)

# Ensure tz-aware UTC without double-localizing
pred["logged_at_utc"] = pd.to_datetime(pred["logged_at_utc"], utc=True, errors="coerce")
pred["asof"]          = pd.to_datetime(pred["asof"],          utc=True, errors="coerce")
act["asof"]           = pd.to_datetime(act["asof"],           utc=True, errors="coerce")

# Join key (by day). If you prefer month-end alignment, switch to .dt.to_period('M').dt.to_timestamp('M')
pred["asof_key"] = pred["asof"].dt.floor("D")
act["asof_key"]  = act["asof"].dt.floor("D")

# Merge predictions with actuals
df = pred.merge(act[["asof_key","actual_return"]], on="asof_key", how="inner")
if df.empty:
    raise SystemExit("No overlapping dates between predictions and actuals after alignment.")

# Errors
df["abs_err"] = (df["prediction"] - df["actual_return"]).abs()
df["sq_err"]  = (df["prediction"] - df["actual_return"])**2

# Per-route summary
summary = (
    df.groupby("route", dropna=False)
      .agg(n=("prediction","size"),
           mae=("abs_err","mean"),
           rmse=("sq_err", lambda s: (s.mean())**0.5))
      .reset_index()
)

OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_SUMMARY, index=False)

# Save pointwise errors for plotting
cols = ["logged_at_utc","asof","asof_key","route","region","prediction","actual_return","abs_err","sq_err"]
df.sort_values(["asof_key","route"]).to_csv(OUT_ERRORS, index=False, columns=[c for c in cols if c in df.columns])

print(f"Wrote {OUT_SUMMARY} and {OUT_ERRORS}")
print(summary.to_string(index=False))
