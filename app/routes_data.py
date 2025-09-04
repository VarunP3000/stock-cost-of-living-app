# app/routes_data.py
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import pandas as pd
import numpy as np


router = APIRouter(tags=["data"])

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

def _first_existing(*names: str) -> Path | None:
    """
    Return the first existing path among:
      data/processed/<name>, data/<name> for each provided name.
    """
    candidates = []
    for name in names:
        candidates += [PROCESSED / name, DATA / name]
    for p in candidates:
        if p.exists():
            return p
    return None

def _require(path: Path | None, err: str) -> Path:
    if path is None:
        raise HTTPException(404, err)
    return path

def _norm_month(df: pd.DataFrame) -> pd.Series:
    if "month" in df.columns:
        s = pd.to_datetime(df["month"], errors="coerce")
    else:
        alt = next((c for c in ["date", "asof", "ds", "Date"] if c in df.columns), None)
        if not alt:
            raise HTTPException(400, "Missing date column (month/date/asof/ds/Date)")
        s = pd.to_datetime(df[alt], errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

def _merged() -> pd.DataFrame | None:
    p = _first_existing("merged_panel.csv")
    if not p:
        return None
    df = pd.read_csv(p)
    df["month"] = _norm_month(df)

    # normalize column names
    if "geo" not in df.columns:
        if "country" in df.columns:
            df = df.rename(columns={"country": "geo"})
        else:
            raise HTTPException(500, f"{p} missing 'geo' (or 'country')")
    if "cpi_yoy" not in df.columns:
        alt = next((c for c in ["cpi", "value", "yoy", "y"] if c in df.columns), None)
        if not alt:
            raise HTTPException(500, f"{p} missing 'cpi_yoy' (or cpi/value/yoy/y)")
        df = df.rename(columns={alt: "cpi_yoy"})
    if "spx_ret" not in df.columns:
        alt = next((c for c in ["actual_return", "return", "y", "value"] if c in df.columns), None)
        if not alt:
            raise HTTPException(500, f"{p} missing 'spx_ret' (or actual_return/return/y/value)")
        df = df.rename(columns={alt: "spx_ret"})

    df["cpi_yoy"] = pd.to_numeric(df["cpi_yoy"], errors="coerce")
    df["spx_ret"] = pd.to_numeric(df["spx_ret"], errors="coerce")
    return df

def _cpi() -> pd.DataFrame:
    # Prefer merged if present (covers both CPI + SPX)
    m = _merged()
    if m is not None:
        return m[["month", "geo", "cpi_yoy"]].copy()

    # Otherwise look for standalone CPI under processed/ or data/
    p = _require(
        _first_existing("spx_cpi_global.csv"),
        f"Missing CPI file. Looked for: {PROCESSED/'spx_cpi_global.csv'} and {DATA/'spx_cpi_global.csv'} (or merged_panel.csv).",
    )
    df = pd.read_csv(p)
    df["month"] = _norm_month(df)
    if "geo" not in df.columns:
        if "country" in df.columns:
            df = df.rename(columns={"country": "geo"})
        else:
            raise HTTPException(400, f"{p} needs geo/country column")
    if "cpi_yoy" not in df.columns:
        alt = next((c for c in ["cpi", "value", "yoy", "y"] if c in df.columns), None)
        if not alt:
            raise HTTPException(400, f"{p} needs CPI value column (cpi_yoy/cpi/value/yoy/y)")
        df = df.rename(columns={alt: "cpi_yoy"})
    df["cpi_yoy"] = pd.to_numeric(df["cpi_yoy"], errors="coerce")
    return df[["month", "geo", "cpi_yoy"]].dropna(subset=["month"])

def _spx() -> pd.DataFrame:
    m = _merged()
    if m is not None:
        out = m[["month", "spx_ret"]].drop_duplicates("month").copy()
        out["spx_ret"] = pd.to_numeric(out["spx_ret"], errors="coerce")
        return out

    p = _require(
        _first_existing("spx_actuals.csv"),
        f"Missing SPX file. Looked for: {PROCESSED/'spx_actuals.csv'} and {DATA/'spx_actuals.csv'} (or merged_panel.csv).",
    )
    df = pd.read_csv(p)
    df["month"] = _norm_month(df)
    v = next((c for c in ["actual_return", "spx_ret", "return", "y", "value"] if c in df.columns), None)
    if not v:
        raise HTTPException(400, f"{p} needs value column (actual_return/spx_ret/return/y/value)")
    df = df.rename(columns={v: "spx_ret"})
    df["spx_ret"] = pd.to_numeric(df["spx_ret"], errors="coerce")
    return df[["month", "spx_ret"]].dropna(subset=["month"]).drop_duplicates("month")

@router.get("/countries")
def countries():
    return sorted(_cpi()["geo"].dropna().unique().tolist())

@router.get("/series")
def series(
    geo: str,
    kind: str = Query(..., pattern="^(cpi_yoy|spx_ret)$")
):
    if kind == "cpi_yoy":
        sub = _cpi().query("geo == @geo").sort_values("month")
        if sub.empty:
            raise HTTPException(404, f"No CPI rows for geo='{geo}'")

        # sanitize: coerce to numeric and remove ±inf to avoid JSON errors
        sub["cpi_yoy"] = pd.to_numeric(sub["cpi_yoy"], errors="coerce")
        sub.loc[~np.isfinite(sub["cpi_yoy"]), "cpi_yoy"] = np.nan
        sub = sub.dropna(subset=["cpi_yoy"])
        if sub.empty:
            raise HTTPException(404, f"Empty series for geo='{geo}', kind='cpi_yoy'")

        return {
            "geo": geo,
            "kind": "cpi_yoy",
            "series": [[m.strftime("%Y-%m-%d"), float(v)] for m, v in sub[["month", "cpi_yoy"]].to_numpy()]
        }

    else:
        spx = _spx().sort_values("month")
        return {
            "geo": geo,
            "kind": "spx_ret",
            "series": [[m.strftime("%Y-%m-%d"), (None if pd.isna(v) else float(v))]
                       for m, v in spx[["month", "spx_ret"]].to_numpy()]
        }
