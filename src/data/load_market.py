# src/features/merge_global.py
from pathlib import Path
import pandas as pd

# ---------- Paths ----------
DATA = Path(__file__).resolve().parents[2] / "data" / "processed"
CPI_PATH = DATA / "cpi_kaggle.csv"          # your processed CPI (tidy: ds, geo, value)
SPX_PATH = DATA / "sp500_monthly.csv"       # your monthly S&P500 (ds, spx_close)
OUT_PATH = DATA / "spx_cpi_global.csv"

# ---------- Helpers ----------
def force_naive(s: pd.Series) -> pd.Series:
    """
    Parse strings like '1990-01-01 00:00:00-05:00' to tz-aware UTC, then drop tz -> naive.
    Works whether input is string, tz-aware, or already naive.
    """
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)

def month_start(s: pd.Series) -> pd.Series:
    """
    Normalize timestamps to the first day of the month at 00:00 (month start).
    Uses Period('M') -> Timestamp().
    """
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def _expand_monthly_ffill(cpi_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    For each geo, build a complete MonthStart index from min->max and forward-fill CPI.
    Expects columns: ['geo', 'ds', 'value'] with ds already month-normalized.
    Returns columns: ['ds', 'geo', 'cpi_index'].
    """
    out = []
    for geo, g in cpi_monthly.groupby("geo", sort=False):
        g = g.sort_values("ds")
        if g.empty:
            continue
        idx = pd.date_range(g["ds"].min(), g["ds"].max(), freq="MS")
        g2 = g.set_index("ds").reindex(idx).rename_axis("ds")
        g2["geo"] = geo
        g2["cpi_index"] = g2["value"].ffill()
        out.append(g2[["geo", "cpi_index"]].reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["ds", "geo", "cpi_index"])

# ---------- Core ----------
def merge_global(start: str = "2000-01-01") -> pd.DataFrame:
    # --- Load sources ---
    cpi = pd.read_csv(CPI_PATH)
    spx = pd.read_csv(SPX_PATH)

    # --- Clean/normalize dates ---
    cpi["ds"] = force_naive(cpi["ds"])
    spx["ds"] = force_naive(spx["ds"])

    # Normalize both to month-start timestamps (no 'MS' arg here)
    cpi["ds"] = month_start(cpi["ds"])
    spx["ds"] = month_start(spx["ds"])

    # Ensure numeric CPI and drop nulls
    cpi["value"] = pd.to_numeric(cpi["value"], errors="coerce")
    cpi = cpi.dropna(subset=["value", "ds", "geo"])

    # Filter to start date (after normalizing to naive month-start)
    start_ts = pd.Timestamp(start)
    cpi = cpi[cpi["ds"] >= start_ts]
    spx = spx[spx["ds"] >= start_ts]

    # Collapse to one row per (geo, ds) BEFORE any expansion
    # (If there are multiple CPI series per country/month, we average them. Change to .first() if preferred.)
    cpi_monthly = (
        cpi.groupby(["geo", "ds"], as_index=False)["value"]
           .mean()
           .sort_values(["geo", "ds"])
    )

    # Expand to complete monthly span per geo and forward-fill gaps
    cpi_full = _expand_monthly_ffill(cpi_monthly)

    # Merge monthly CPI with monthly SPX (same SPX across all geos)
    spx_m = spx.drop_duplicates(subset=["ds"])[["ds", "spx_close"]].sort_values("ds")
    df = cpi_full.merge(spx_m, on="ds", how="left")

    # Final ordering + features
    df = df.sort_values(["geo", "ds"]).reset_index(drop=True)
    df["spx_ret_1m"] = df["spx_close"].pct_change(1)                # same across geos
    df["cpi_mom"]    = df.groupby("geo")["cpi_index"].pct_change(1)
    df["cpi_yoy"]    = df.groupby("geo")["cpi_index"].pct_change(12)

    # Safety: ensure no duplicated (geo, ds) remain
    df = df.drop_duplicates(subset=["geo", "ds"])
    return df

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = merge_global()
    df.to_csv(OUT_PATH, index=False)
    n_geos = df["geo"].nunique() if "geo" in df.columns else 0
    print(
        f"Wrote {OUT_PATH} with {len(df)} rows • {n_geos} geos • "
        f"{df['ds'].min().date()} → {df['ds'].max().date()}"
    )

if __name__ == "__main__":
    main()
