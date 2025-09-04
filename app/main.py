# app/main.py
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import numpy as np
import logging

app = FastAPI(title="Inflation → SPX API (adaptive)")
logging.basicConfig(level=logging.INFO)

# CORS for Next.js dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# Import routers individually and track which ones are present
has_data = has_corr = has_forecast = False

try:
    from app.routes_data import router as data_router  # /countries, /series
    app.include_router(data_router)
    has_data = True
    logging.info("Mounted routes_data router")
except Exception as e:
    logging.info("routes_data not mounted: %s", e)

try:
    from app.routes_correlations import router as corr_router  # /correlations
    app.include_router(corr_router)
    has_corr = True
    logging.info("Mounted routes_correlations router")
except Exception as e:
    logging.info("routes_correlations not mounted: %s", e)

try:
    from app.routes_forecast import router as forecast_router  # /forecast/*
    app.include_router(forecast_router)
    has_forecast = True
    logging.info("Mounted routes_forecast router")
except Exception as e:
    logging.info("routes_forecast not mounted: %s", e)

# NEW: Ensemble routes (custom weights: past metrics + future forecast)
try:
    from app.routes_ensemble import router as ensemble_router  # /ensemble/*
    app.include_router(ensemble_router)
    logging.info("Mounted routes_ensemble router")
except Exception as e:
    logging.info("routes_ensemble not mounted: %s", e)

# ---------- Fallback helpers reading your CSVs ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Prefer the cleaned, processed panel; fallback to the unclean processed panel
CPI_PANEL_PRIMARY = DATA_DIR / "processed" / "spx_cpi_global_clean.csv"
CPI_PANEL_FALLBACK = DATA_DIR / "processed" / "spx_cpi_global.csv"
SPX_FILE = DATA_DIR / "spx_actuals.csv"   # this exists in your tree

def _normalize_month_col(df: pd.DataFrame, prefer_col: str = "month") -> pd.Series:
    """Return a pandas datetime Series (Month start) from any common date-like column."""
    if prefer_col in df.columns:
        s = pd.to_datetime(df[prefer_col], errors="coerce")
    else:
        alt = next((c for c in ["date", "asof", "ds", "Date"] if c in df.columns), None)
        if not alt:
            raise HTTPException(400, "Missing month/date column (expected one of: month, date, asof, ds, Date)")
        s = pd.to_datetime(df[alt], errors="coerce")
    s = s.dt.to_period("M").dt.to_timestamp()
    return s

def _load_cpi_panel() -> pd.DataFrame:
    """Load the CPI/SPX panel used by /countries and CPI series kinds."""
    path = None
    if CPI_PANEL_PRIMARY.exists():
        path = CPI_PANEL_PRIMARY
    elif CPI_PANEL_FALLBACK.exists():
        path = CPI_PANEL_FALLBACK
    else:
        raise HTTPException(404, f"Missing processed CPI panel at {CPI_PANEL_PRIMARY} or {CPI_PANEL_FALLBACK}")

    df = pd.read_csv(path)
    # normalize month
    df["month"] = _normalize_month_col(df)
    # required columns
    required_any = {"geo", "cpi_yoy"}
    if not required_any.issubset(df.columns):
        raise HTTPException(500, f"CPI panel missing required columns; found {list(df.columns)}")
    # numeric coercions (be liberal)
    for col in ["cpi_index", "cpi_yoy", "cpi_mom", "spx_close", "spx_ret_1m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _load_spx() -> pd.DataFrame:
    if not SPX_FILE.exists():
        raise HTTPException(404, f"Missing {SPX_FILE}")
    df = pd.read_csv(SPX_FILE)
    df["month"] = _normalize_month_col(df)
    vcol = next((c for c in ["actual_return","spx_ret","return","y","value"] if c in df.columns), None)
    if not vcol:
        raise HTTPException(400, "SPX file needs value column (actual_return/spx_ret/return/y/value)")
    df = df.rename(columns={vcol: "spx_ret"})
    df = df.dropna(subset=["month"])
    df["spx_ret"] = pd.to_numeric(df["spx_ret"], errors="coerce")
    df = df.sort_values("month").drop_duplicates("month")
    return df[["month","spx_ret"]]

# ---------- Fallback DATA routes (if /countries & /series missing) ----------
if not has_data:
    data_fallback = APIRouter(tags=["data_fallback"])

    @data_fallback.get("/countries")
    def countries():
        panel = _load_cpi_panel()
        geos = sorted(panel["geo"].dropna().astype(str).unique().tolist())
        return geos  # keep your original shape (list of strings)

    @data_fallback.get("/series")
    def series(
        geo: str,
        kind: str = Query(..., pattern="^(cpi_yoy|cpi_mom|cpi_index|spx_close|spx_ret_1m|spx_ret)$")
    ):
        """Return {"geo","kind","series":[[YYYY-MM-DD, value], ...]} matching your frontend contract."""
        k = kind.strip().lower()
        logging.info("GET /series geo=%s kind=%s", geo, k)

        if k == "spx_ret":
            spx = _load_spx().sort_values("month")
            # no geo filter for the index-only series
            series = [
                [m.strftime("%Y-%m-%d"), (None if pd.isna(v) else float(v))]
                for m, v in zip(spx["month"], spx["spx_ret"])
            ]
            if not series:
                raise HTTPException(404, "Empty SPX series")
            return {"geo": geo, "kind": "spx_ret", "series": series}

        # CPI panel kinds (country-specific)
        panel = _load_cpi_panel()
        if geo not in set(panel["geo"].astype(str).unique()):
            # case-insensitive fallback
            sub = panel[panel["geo"].str.lower() == geo.lower()]
        else:
            sub = panel[panel["geo"] == geo]

        if sub.empty:
            raise HTTPException(404, f"No rows for geo='{geo}'")

        col_map = {
            "cpi_yoy": "cpi_yoy",
            "cpi_mom": "cpi_mom",
            "cpi_index": "cpi_index",
            "spx_close": "spx_close",
            "spx_ret_1m": "spx_ret_1m",
        }
        col = col_map.get(k)
        if not col or col not in sub.columns:
            raise HTTPException(400, f"Unsupported kind='{kind}' for CPI panel")

        sub = sub.sort_values("month")
        # only non-null rows for the requested column
        sub = sub.dropna(subset=[col])
        if sub.empty:
            raise HTTPException(404, f"Empty series for geo='{geo}', kind='{kind}'")

        series = [
            [m.strftime("%Y-%m-%d"), (None if pd.isna(v) else float(v))]
            for m, v in zip(sub["month"], sub[col])
        ]
        return {"geo": geo, "kind": k, "series": series}

    app.include_router(data_fallback)

# ---------- Fallback CORRELATIONS route (if missing) ----------
if not has_corr:
    corr_fallback = APIRouter(tags=["correlations_fallback"])

    @corr_fallback.get("/correlations")
    def correlations(window_months: int = Query(36, ge=6, le=240)):
        panel = _load_cpi_panel()
        spx = _load_spx().set_index("month")["spx_ret"].sort_index()

        wide = panel.pivot_table(index="month", columns="geo", values="cpi_yoy", aggfunc="mean").sort_index()
        idx = wide.index.intersection(spx.index)
        wide = wide.loc[idx]
        y = spx.loc[idx].to_numpy(dtype=float)

        countries = list(wide.columns)
        dates = [d.strftime("%Y-%m-%d") for d in idx]

        rows = []
        for start in range(0, len(idx) - window_months + 1):
            end = start + window_months
            y_win = y[start:end]
            row = []
            for c in countries:
                x = wide[c].iloc[start:end].to_numpy(dtype=float)
                mask = ~np.isnan(x) & ~np.isnan(y_win)
                if mask.sum() < 2:
                    row.append(None)
                else:
                    xv, yv = x[mask], y_win[mask]
                    xmu, ymu = xv.mean(), yv.mean()
                    xs = np.sqrt(((xv - xmu) ** 2).sum())
                    ys = np.sqrt(((yv - ymu) ** 2).sum())
                    if xs == 0 or ys == 0:
                        row.append(None)
                    else:
                        cov = ((xv - xmu) * (yv - ymu)).sum()
                        row.append(float(cov / (xs * ys)))
            rows.append(row)

        dates = dates[window_months - 1:]
        return {"window_months": window_months, "countries": countries, "dates": dates, "matrix": rows}

    app.include_router(corr_fallback)

    # NEW: realtime inference routes
try:
    from app.routes_realtime import router as realtime_router  # /realtime/*
    app.include_router(realtime_router)
    logging.info("Mounted routes_realtime router")
except Exception as e:
    logging.info("routes_realtime not mounted: %s", e)


# NOTE: Removed stray include_router(...) calls that referenced undefined routers.
