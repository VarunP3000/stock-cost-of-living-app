import csv, os, json
from datetime import datetime

LOG_PATH = os.environ.get("PRED_LOG_PATH", "data/prediction_log.csv")

def log_prediction(route: str, response: dict, features: list[float] | None = None):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts","route","response_json","features_json"])
        w.writerow([datetime.utcnow().isoformat(), route,
                    json.dumps(response), json.dumps(features or [])])
