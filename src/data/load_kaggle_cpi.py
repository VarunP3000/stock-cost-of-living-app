from pathlib import Path
import pandas as pd
import re

DATA = Path(__file__).resolve().parents[2] / "data" / "external" / "kaggle" / "cpi.csv"
OUT  = Path(__file__).resolve().parents[2] / "data" / "processed" / "cpi_kaggle.csv"

DATE_PATTERNS = [
    re.compile(r"^\d{4}$"),       # e.g., 1913
    re.compile(r"^\d{4}-\d{2}$"), # e.g., 1913-01
]

def _dedup_cols(cols):
    """Make duplicate column names unique: 'decimals', 'decimals.1', 'decimals.2', ..."""
    seen = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def _is_date_col(name: str) -> bool:
    n = str(name).strip().lower()
    return any(p.match(n) for p in DATE_PATTERNS)

def load_kaggle_cpi() -> pd.DataFrame:
    # read csv and ensure unique column names
    df = pd.read_csv(DATA, dtype=str)  # read as str first; we’ll cast later
    df.columns = _dedup_cols(df.columns)
    # normalize column names
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))

    # geo column candidates (your file shows ref_area / reference_area)
    if "reference_area" in df.columns:
        geo_col = "reference_area"
    elif "ref_area" in df.columns:
        geo_col = "ref_area"
    else:
        # last resort — try common labels
        for cand in ("country", "location", "economy", "name"):
            if cand in df.columns:
                geo_col = cand
                break
        else:
            raise ValueError(f"No country/location column found. First columns: {list(df.columns)[:20]}")

    # date columns (wide year / year-month headers)
    date_cols = [c for c in df.columns if _is_date_col(c)]
    if not date_cols:
        raise ValueError(f"No date columns like YYYY or YYYY-MM detected. First columns: {list(df.columns)[:30]}")

    # keep the melt IDs minimal to avoid weird dtype issues
    id_cols = [geo_col]

    # melt wide → long
    long = df.melt(id_vars=id_cols, value_vars=date_cols, var_name="ds_raw", value_name="value")

    # parse datetime
    is_ym = long["ds_raw"].str.match(r"^\d{4}-\d{2}$", na=False)
    is_y  = long["ds_raw"].str.match(r"^\d{4}$",       na=False)

    long.loc[is_ym, "ds"] = pd.to_datetime(long.loc[is_ym, "ds_raw"], format="%Y-%m", errors="coerce")
    long.loc[is_y,  "ds"] = pd.to_datetime(long.loc[is_y,  "ds_raw"] + "-01-01", format="%Y-%m-%d", errors="coerce")

    out = long.dropna(subset=["ds"]).copy()
    out = out.rename(columns={geo_col: "geo"})[["ds", "geo", "value"]]
    # cast numeric; drop rows that aren’t numbers
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])

    out["metric"] = "CPI_Index"
    out = out[["ds", "geo", "metric", "value"]].sort_values(["geo", "ds"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(
        f"Wrote {OUT} with {len(out)} rows, {out['geo'].nunique()} geos, "
        f"{out['ds'].min().date()} → {out['ds'].max().date()}"
    )
    return out

if __name__ == "__main__":
    load_kaggle_cpi()
