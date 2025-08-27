from pathlib import Path
import pandas as pd

IN  = Path(__file__).resolve().parents[2] / "data" / "processed" / "sp500.csv"
OUT = Path(__file__).resolve().parents[2] / "data" / "processed" / "sp500_monthly.csv"

def main():
    # parse 'ds' as datetime
    df = pd.read_csv(IN, parse_dates=["ds"])
    if df["ds"].dtype.name != "datetime64[ns]":
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # resample to month start, keep last close of each month
    spx_m = (
        df.set_index("ds")["close"]
          .resample("MS").last()
          .rename("spx_close")
          .to_frame()
          .reset_index()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spx_m.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(spx_m)} rows: {spx_m['ds'].min().date()} → {spx_m['ds'].max().date()}")

if __name__ == "__main__":
    main()

