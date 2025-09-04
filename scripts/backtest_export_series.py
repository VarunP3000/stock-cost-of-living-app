#!/usr/bin/env python
"""
Export per-month prediction vs actual series for a given MODEL (ensemble|ridge|elasticnet|gb).

Usage:
  python scripts/backtest_export_series.py --model ensemble [--verbose]
Output:
  reports/backtest_series_forecast_<model>.csv
Columns:
  month, prediction, actual, p10, p90
"""
from __future__ import annotations
import argparse, csv, json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
PRED_LOG = ROOT / "data" / "prediction_log.csv"
ACTUALS  = ROOT / "data" / "spx_actuals.csv"
REPORTS  = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _month_key_dt(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}-01"

def _try_parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    s = s.strip().replace("Z", "")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _match_model(route_val: str | None, model: str) -> bool:
    if not route_val: return False
    r = route_val.strip().lower()
    m = model.lower()
    return (
        r == m or
        r.endswith("/" + m) or
        r.endswith("_" + m) or
        r == f"/forecast/{m}" or
        r.endswith(f"/forecast/{m}")
    )

def _extract_asof(resp: dict, fallback_ts: Optional[datetime]) -> Optional[datetime]:
    for k in ("asof", "as_of", "date", "timestamp"):
        v = resp.get(k)
        dt = _try_parse_dt(v if isinstance(v, str) else None)
        if dt: return dt
    return fallback_ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ensemble","ridge","elasticnet","gb"], default="ensemble")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    model = args.model
    verbose = args.verbose

    preds = _read_csv(PRED_LOG)
    acts  = _read_csv(ACTUALS)
    if not preds:
        raise SystemExit(f"Missing or empty {PRED_LOG}")
    if not acts:
        raise SystemExit(f"Missing or empty {ACTUALS}")

    # Gather route stats
    distinct_routes = {}
    for r in preds:
        route = (r.get("route") or "").strip()
        distinct_routes[route] = distinct_routes.get(route, 0) + 1

    # ---- latest prediction per month for the chosen model ----
    matched = [r for r in preds if _match_model(r.get("route"), model)]
    if verbose:
        print(f"rows total={len(preds)}, matched_model={len(matched)}; routes={distinct_routes}")

    per_month: Dict[str, Dict] = {}
    for r in matched:
        ts = _try_parse_dt(r.get("timestamp"))
        # parse response_json (be robust to bad JSON)
        raw = r.get("response_json") or "{}"
        try:
            resp = json.loads(raw)
        except Exception:
            if verbose:
                print("WARN: bad JSON in response_json; skipping row")
            continue
        # prediction / bands
        pred = resp.get("prediction")
        p10  = resp.get("p10")
        p90  = resp.get("p90")
        # asof (prefer response_json.asof)
        asof = _extract_asof(resp, ts)
        if not asof:
            if verbose:
                print("WARN: no asof/timestamp; skipping row")
            continue

        mkey = _month_key_dt(asof)
        existing = per_month.get(mkey)
        if existing is None or (ts and existing["timestamp"] and ts > existing["timestamp"]) or (existing["timestamp"] is None and ts is not None):
            per_month[mkey] = {
                "timestamp": ts,
                "prediction": pred,
                "p10": p10,
                "p90": p90,
            }

    if not per_month:
        raise SystemExit(
            f"No usable rows for model='{model}'.\n"
            f"Route counts: {distinct_routes}\n"
            f"Tip: ensure prediction_log.csv has rows with route like '/forecast/{model}' "
            "and response_json containing 'asof' and 'prediction'."
        )

    # ---- actuals columns detection ----
    date_key = None
    val_key  = None
    if acts:
        keys = set(acts[0].keys())
        for k in ("asof","date","month","ds","Date"):  # date column
            if k in keys and date_key is None: date_key = k
        for k in ("actual_return","spx_ret","return","y","value"):  # value column
            if k in keys and val_key is None: val_key = k
    if not date_key or not val_key:
        raise SystemExit(
            "spx_actuals.csv missing date/value columns. "
            "Need one of (asof,date,month,ds,Date) and one of (actual_return,spx_ret,return,y,value)."
        )

    actuals: Dict[str, Optional[float]] = {}
    for r in acts:
        dt = _try_parse_dt(r.get(date_key))
        if not dt: continue
        mkey = _month_key_dt(dt)
        v = r.get(val_key)
        actuals[mkey] = None if v in (None,"") else float(v)

    # ---- write output ----
    out = REPORTS / f"backtest_series_forecast_{model}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["month","prediction","actual","p10","p90"])
        w.writeheader()
        for m in sorted(per_month.keys()):
            row = per_month[m]
            w.writerow({
                "month": m,
                "prediction": row.get("prediction"),
                "actual": actuals.get(m),
                "p10": row.get("p10"),
                "p90": row.get("p90"),
            })

    if verbose:
        print(f"Wrote {out} with {len(per_month)} months; "
              f"first_month={min(per_month.keys()) if per_month else 'NA'}, "
              f"last_month={max(per_month.keys()) if per_month else 'NA'}")
    else:
        print(f"Wrote {out} with {len(per_month)} rows")

if __name__ == "__main__":
    main()
