from __future__ import annotations
from pathlib import Path
from datetime import date
import requests
import pandas as pd

SERIES_ID = "CUSR0000SA0"  # CPI-U All items, SA
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT = DATA_DIR / "processed" / "cpi.csv"

def fetch_cpi(start_year: int = 1990, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or date.today().year
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {"seriesid": [SERIES_ID], "startyear": str(start_year), "endyear": str(end_year)}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {j}")
    rows = j["Results"]["series"][0]["data"]
    # to DataFrame
    df = pd.DataFrame(rows)
    # year, period ("M01".."M12"), value (string) → timestamp, float
    df = df[df["period"].str.startswith("M")].copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["period"].str[1:].astype(int)
    df["value"] = df["value"].astype(float)
    df["ds"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values("ds", ascending=True)[["ds","value"]].rename(columns={"value":"cpi_sa"})
    return df

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    df = fetch_cpi()
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} with {len(df)} rows")

if __name__ == "__main__":
    main()
