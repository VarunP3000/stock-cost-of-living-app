from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.data.load_market import load_sp500_history

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def spx_monthly(start="1990-01-01") -> pd.DataFrame:
    spx = load_sp500_history(start=start)
    spx_m = (spx.set_index("ds")["close"]
                .resample("MS").last()
                .rename("spx_close")
                .to_frame()
                .reset_index())
    return spx_m

def load_cpi_processed() -> pd.DataFrame:
    cpi = pd.read_csv(DATA_DIR / "processed" / "cpi.csv", parse_dates=["ds"])
    return cpi

def merge_spx_cpi(start="1990-01-01") -> pd.DataFrame:
    spx_m = spx_monthly(start)
    cpi = load_cpi_processed()
    df = pd.merge(spx_m, cpi, on="ds", how="inner").sort_values("ds")
    # features
    df["spx_ret_1m"]  = df["spx_close"].pct_change(1)
    df["spx_ret_12m"] = df["spx_close"].pct_change(12)
    df["cpi_yoy"]     = df["cpi_sa"].pct_change(12)
    df["cpi_mom"]     = df["cpi_sa"].pct_change(1)
    return df

if __name__ == "__main__":
    out = DATA_DIR / "processed" / "spx_cpi_monthly.csv"
    merged = merge_spx_cpi()
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {out} with {len(merged)} rows")
