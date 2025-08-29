import json, os
from datetime import datetime
from typing import Any, Dict

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "models/artifacts")

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def load_metrics(artifact_name: str) -> Dict[str, Any]:
    base = os.path.splitext(artifact_name)[0]
    path = os.path.join(ARTIFACTS_DIR, f"{base}.metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def try_get_coefficients(model):
    try:
        coefs = getattr(model, "coef_", None)
        if coefs is None:
            return None
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            return {"values": [float(c) for c in coefs]}
        return {str(n): float(c) for n, c in zip(names, coefs)}
    except Exception:
        return None

# ---- loaders ----
def load_model(artifact_name: str):
    """Load a model artifact from ARTIFACTS_DIR using joblib."""
    import joblib
    path = os.path.join(ARTIFACTS_DIR, artifact_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path} (ARTIFACTS_DIR={ARTIFACTS_DIR})")
    return joblib.load(path)

def build_latest_feature_row(region: str | None = None):
    """
    Delegate to your real feature builder if present; otherwise raise a clear error.
    This import happens at CALL time (not import time) to avoid breaking the server boot.
    """
    try:
        from app.api.feature_builder import build_latest_feature_row as impl  # your real builder
        return impl(region=region)
    except Exception as e:
        raise RuntimeError(
            "build_latest_feature_row is not implemented. "
            "Create app/api/feature_builder.py with a build_latest_feature_row(region=None) "
            "that returns a 2D array (shape 1 x n_features) aligned with training."
        ) from e
