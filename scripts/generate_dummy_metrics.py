import os, json, time
from pathlib import Path

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "models/artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Default stub metrics; update later with real backtest results
DEFAULT = {
    "rmse": None,          # fill with float later
    "mae": None,           # fill with float later
    "r2": None,            # fill with float later
    "samples": 0,          # backtest sample size
    "horizon": "1m",       # forecast horizon
    "trained_on": None,    # e.g., "up to 2025-08"
    "created_at": None,    # script fill
    "notes": "placeholder metrics; replace after backtest"
}

def ensure_metrics(pkl: Path):
    base = pkl.with_suffix("")           # drop .pkl
    metrics_path = base.with_suffix(".metrics.json")
    if metrics_path.exists():
        return False
    m = DEFAULT.copy()
    m["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # heuristic: attach model type from filename
    m["model"] = pkl.name.replace(".pkl","")
    metrics_path.write_text(json.dumps(m, indent=2))
    return True

made = 0
for f in sorted(ARTIFACTS_DIR.glob("*.pkl")):
    if ensure_metrics(f):
        print("Wrote:", f.with_suffix(".metrics.json").name)
        made += 1

print(f"Done. Created {made} metrics files in {ARTIFACTS_DIR}")
