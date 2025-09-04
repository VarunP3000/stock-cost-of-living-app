# scripts/update_artifact_metrics.py
import json, pathlib, time
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "models" / "artifacts"
REPORTS = ROOT / "reports"
SUMMARY = REPORTS / "backtest_summary.csv"

if not SUMMARY.exists():
    raise SystemExit("Missing reports/backtest_summary.csv. Run backtest_enriched.py first.")

summary = pd.read_csv(SUMMARY)

# Map API routes -> artifact filename stems (adjust if your names differ)
ROUTE_TO_ARTIFACT = {
    "/forecast/ridge":        "ridge_spx",
    "/forecast/elasticnet":   "elasticnet_spx",
    "/forecast/gb":           "gb_spx",
    "/forecast/ensemble":     "ensemble_spx",
    "/forecast/quantiles":    "quantiles_spx",
    "/forecast/directional":  "directional_spx",
    # Example regionals (only if you have these artifacts):
    "/forecast/regional":     "ridge_spx_regional"
}

def write_metrics(artifact_stem, metrics: dict):
    metrics_path = ART / f"{artifact_stem}.metrics.json"
    payload = {
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "samples": int(metrics.get("n", 0)),
        "trained_on": metrics.get("asof_min"),
        "evaluated_to": metrics.get("asof_max"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {metrics_path}")

for _, row in summary.iterrows():
    route = row["route"]
    art = ROUTE_TO_ARTIFACT.get(route)
    if art:
        write_metrics(art, row.to_dict())

print("Metrics update complete.")
