# app/routes_correlations.py
from fastapi import APIRouter, Query
from datetime import date
from typing import List, Tuple
import math, random, hashlib

router = APIRouter(tags=["correlations"])

# ---------- demo data helpers ----------
def _demo_dates(months: int = 120) -> List[str]:
    today = date.today().replace(day=1)
    out: List[str] = []
    y, m = today.year, today.month
    for i in range(months):
        mm = m - (months - 1 - i)
        yy = y
        while mm <= 0:
            mm += 12
            yy -= 1
        out.append(f"{yy:04d}-{mm:02d}-01")
    return out

def _seed_from_geo(geo: str) -> int:
    h = hashlib.sha256(geo.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _demo_spx_series() -> List[Tuple[str, float]]:
    random.seed(42)
    ds = _demo_dates(120)
    vals = [0.005 + random.gauss(0, 0.03) for _ in ds]
    return list(zip(ds, vals))

def _demo_cpi_series(geo: str) -> List[Tuple[str, float]]:
    random.seed(_seed_from_geo(geo))
    ds = _demo_dates(120)
    base = 0.02 + (_seed_from_geo(geo) % 3000) / 100000  # 2.00%–4.99%
    phase = (_seed_from_geo(geo) % 12) / 12 * 2 * math.pi
    vals = []
    for i in range(len(ds)):
        vals.append(base + 0.01 * math.sin(phase + i * (2 * math.pi / 18)) + random.gauss(0, 0.0025))
    return list(zip(ds, vals))

def _countries_fallback() -> List[str]:
    return [
        "United States","Canada","United Kingdom","Germany","France","Japan","Brazil",
        "India","Australia","Spain","Algeria","Argentina","Thailand","Mexico","Italy",
    ]

# ---------- math helpers ----------
def _pearson(xs: List[float], ys: List[float]) -> float | None:
    n = len(xs)
    if n == 0 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy)

@router.get("/correlations")
def correlations(window_months: int = Query(36, ge=6, le=180)):
    """
    Rolling correlation between SPX monthly returns (global) and each country's CPI YoY.
    Returns:
      {
        "window_months": int,
        "countries": [str...],
        "dates": [YYYY-MM-DD...],
        "matrix": [[float|null]*len(countries) for each date]
      }
    """
    # Base index / SPX
    spx = _demo_spx_series()
    dates = [d for d, _ in spx]
    spx_vals = [v for _, v in spx]

    countries = _countries_fallback()

    # Build CPI per country aligned on dates
    cpi_by_country: dict[str, List[float]] = {}
    for geo in countries:
        cpi = _demo_cpi_series(geo)
        # they’re already aligned by construction
        cpi_by_country[geo] = [v for _, v in cpi]

    # For each date index, compute corr over the last window
    # Build matrix as dates x countries
    matrix: List[List[float | None]] = []
    for t in range(len(dates)):
        row: List[float | None] = []
        if t + 1 < window_months:
            # not enough data to compute a full window
            row = [None] * len(countries)
        else:
            lo = t + 1 - window_months
            spx_win = spx_vals[lo : t + 1]
            for geo in countries:
                cpi_win = cpi_by_country[geo][lo : t + 1]
                r = _pearson(cpi_win, spx_win)
                row.append(None if r is None or math.isnan(r) else r)
        matrix.append(row)

    return {
        "window_months": window_months,
        "countries": countries,
        "dates": dates,
        "matrix": matrix,
    }
