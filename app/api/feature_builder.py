import numpy as np
import pandas as pd
from typing import Optional
from app.api.utils import load_model, ARTIFACTS_DIR

GLOBAL_REF = "ridge_spx_v1.pkl"

def _feature_names_and_count(artifact: str):
    m = load_model(artifact)
    names = getattr(m, "feature_names_in_", None)
    if names is not None:
        return list(map(str, names)), len(names)
    # Fallback via coef length; make synthetic names
    coef = getattr(m, "coef_", None)
    if coef is not None:
        n = int(len(coef))
        return [f"f{i}" for i in range(n)], n
    n = getattr(m, "n_features_in_", None)
    if n is not None:
        n = int(n)
        return [f"f{i}" for i in range(n)], n
    raise RuntimeError(
        f"Could not infer features from {artifact} in {ARTIFACTS_DIR}. "
        "Model should expose feature_names_in_, coef_ or n_features_in_."
    )

def build_latest_feature_row(region: Optional[str] = None):
    """
    TEMPORARY builder: returns a 1 x n_features DataFrame of zeros
    with the correct column names so sklearn stops warning.
    Replace with your real feature pipeline when ready.
    """
    artifact = GLOBAL_REF
    if region:
        cand = f"ridge_spx_{region}.pkl"
        try:
            names, n = _feature_names_and_count(cand)
            return pd.DataFrame([np.zeros(n, dtype=float)], columns=names)
        except Exception:
            pass  # fall back to global
    names, n = _feature_names_and_count(GLOBAL_REF)
    return pd.DataFrame([np.zeros(n, dtype=float)], columns=names)
