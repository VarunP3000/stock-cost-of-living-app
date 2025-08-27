from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "processed"
CPI_PATH = DATA / "cpi_kaggle.csv"
SPX_PATH = DATA / "sp500_monthly.csv"
OUT_PATH = DATA / "spx_cpi_global.csv"

def force_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)

def merge_global(start: str = "2000-01-01") -> pd.DataFrame:
    cpi = pd.read_csv(CPI_PATH)
    spx = pd.read_csv(SPX_PATH)

    cpi["ds"] = force_naive(cpi["ds"]).dt.to_period("M").dt.to_timestamp()
    spx["ds"] = force_naive(spx["ds"]).dt.to_period("M").dt.to_timestamp()

    cpi["value"] = pd.to_numeric(cpi["value"], errors="coerce")
    cpi = cpi.dropna(subset=["value", "ds", "geo"]).drop_duplicates(["geo","ds"])
    spx = spx.drop_duplicates(subset=["ds"])

    start_ts = pd.Timestamp(start)
    cpi = cpi[cpi["ds"] >= start_ts]
    spx = spx[spx["ds"] >= start_ts]

    # collapse duplicates per geo/month
    cpi_monthly = (
        cpi.groupby(["geo","ds"], as_index=False)["value"].mean()
    ).sort_values(["geo","ds"])

    # merge
    df = cpi_monthly.merge(spx, on="ds", how="left")
    df = df.rename(columns={"value":"cpi_index"})
    df = df.sort_values(["geo","ds"]).reset_index(drop=True)

    # features
    df["spx_ret_1m"] = df["spx_close"].pct_change(1)
    df["cpi_mom"]    = df.groupby("geo")["cpi_index"].pct_change(1)
    df["cpi_yoy"]    = df.groupby("geo")["cpi_index"].pct_change(12)

    return df

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = merge_global()
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote {OUT_PATH} with {len(df)} rows, "
          f"{df['geo'].nunique()} geos, "
          f"{df['ds'].min().date()} → {df['ds'].max().date()}")

if __name__ == "__main__":
    main()
