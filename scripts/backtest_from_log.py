import csv, json
from pathlib import Path
from datetime import datetime
import pandas as pd

LOG_PATH = Path("data/prediction_log.csv")
OUT_PATH = Path("data/predictions_tidy.csv")

if not LOG_PATH.exists():
    raise SystemExit(f"Log not found: {LOG_PATH}. Hit endpoints first to generate logs.")

rows = []
with LOG_PATH.open() as f:
    r = csv.DictReader(f)
    for line in r:
        ts = line["ts"]
        route = line["route"]
        try:
            resp = json.loads(line["response_json"])
        except Exception:
            continue
        # normalize ensemble vs regional vs others
        asof = resp.get("asof")
        pred = resp.get("prediction")
        region = resp.get("region")
        weights = resp.get("weights_used")
        components = resp.get("components")

        rows.append({
            "logged_at_utc": ts,
            "asof": asof,
            "route": route,
            "region": region,
            "prediction": pred,
            "components_json": json.dumps(components) if components else None,
            "weights_json": json.dumps(weights) if weights else None,
        })

df = pd.DataFrame(rows)

# coerce times
for col in ["logged_at_utc", "asof"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# sort and write
df = df.sort_values(["logged_at_utc","route"]).reset_index(drop=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows -> {OUT_PATH}")
print(df.tail(5).to_string(index=False))
