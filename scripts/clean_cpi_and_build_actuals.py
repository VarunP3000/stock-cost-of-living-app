#!/usr/bin/env python
# scripts/clean_cpi_and_build_actuals.py
"""
Prepares:
  - data/processed/spx_cpi_global_clean.csv  (ffill/bfill, drop all-NaN cols)
  - data/spx_actuals.csv                     (asof, actual_return) from sp500_monthly.csv
"""

import pathlib
import sys
import pandas as pd


def infer_date_col(df: pd.DataFrame) -> str:
    best, good_best = None, -1
    for c in df.columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce")
            good = dt.notna().sum()
            if good > good_best and good > 3:
                best, good_best = c, good
        except Exception:
            pass
    if best is None:
        raise ValueError("Could not infer a date/asof column.")
    return best


def prepare_cpi():
    src = pathlib.Path("data/processed/spx_cpi_global.csv")
    if not src.exists():
        sys.exit(f"Missing {src} — please check the path.")
    cpi = pd.read_csv(src)
    dc = infer_date_col(cpi)
    cpi["asof"] = pd.to_datetime(cpi[dc], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if dc != "asof":
        cpi = cpi.drop(columns=[dc], errors="ignore")
    cpi = cpi.dropna(subset=["asof"]).sort_values("asof")
    cols = [c for c in cpi.columns if c != "asof"]
    # ffill/bfill across time
    cpi[cols] = cpi[cols].ffill().bfill()
    # drop all-NaN columns (just in case)
    keep = ["asof"] + [c for c in cols if cpi[c].notna().any()]
    cpi = cpi[keep]
    out = pathlib.Path("data/processed/spx_cpi_global_clean.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cpi.to_csv(out, index=False)
    print(f"Wrote {out} shape={cpi.shape}")


def prepare_actuals():
    src = pathlib.Path("data/processed/sp500_monthly.csv")
    if not src.exists():
        sys.exit(f"Missing {src} — please check the path.")
    df = pd.read_csv(src)

    # infer date col
    dc = infer_date_col(df)
    df["asof"] = pd.to_datetime(df[dc], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["asof"]).sort_values("asof")

    # choose price-like numeric column
    numeric = []
    for c in df.columns:
        if c == "asof":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 6 and s.var(skipna=True) > 0:
            numeric.append((c.lower(), c, s.notna().sum()))
    if not numeric:
        raise ValueError("No numeric price-like column found in sp500_monthly.csv")
    # prioritize names
    priority = ["adj", "adjusted", "close", "price", "index", "spx", "sp500", "value", "level"]
    def rank(name): 
        for i,k in enumerate(priority):
            if k in name:
                return i
        return 99
    numeric.sort(key=lambda t: (rank(t[0]), -t[2]))
    price_col = numeric[0][1]

    df["_p"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["_p"])

    # next-month return
    df["actual_return"] = df["_p"].pct_change().shift(-1)
    out = df.loc[df["actual_return"].notna(), ["asof", "actual_return"]].drop_duplicates("asof")
    outp = pathlib.Path("data/spx_actuals.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"Wrote {outp} rows={len(out)} range={out['asof'].min()}→{out['asof'].max()}")


def main():
    prepare_cpi()
    prepare_actuals()


if __name__ == "__main__":
    main()
