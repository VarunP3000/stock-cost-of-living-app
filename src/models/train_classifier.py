from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

DATA = Path("data/processed")
ART  = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA/"spx_cpi_global.csv", parse_dates=["ds"]).sort_values(["geo","ds"])

# ===== CPI panel (time x geo) and robust cleaning =====
panel = (df.pivot_table(index="ds", columns="geo", values="cpi_yoy")
           .sort_index())

# Replace infs, clip extreme values, forward-fill within each geo, then fill remaining with column medians
panel = panel.replace([np.inf, -np.inf], np.nan)
panel = panel.clip(lower=-100, upper=100)  # CPI YoY should never be this extreme; acts as a guardrail
panel = panel.ffill().dropna(how="all", axis=1)  # drop geos with no data at all
panel = panel.fillna(panel.median(numeric_only=True))
# Final safety check
assert np.isfinite(panel.to_numpy()).all(), "Panel still has non-finite values."

# ===== PCA global factors =====
n_factors = 3
pca = PCA(n_components=n_factors)
fac = pd.DataFrame(pca.fit_transform(panel), index=panel.index,
                   columns=[f"cpi_fac{i+1}" for i in range(n_factors)])

# ===== Market context features =====
spx = (df[["ds","spx_ret_1m"]].drop_duplicates("ds")
         .set_index("ds").sort_index())
# basic cleaning on spx
spx = spx.replace([np.inf, -np.inf], np.nan)
spx = spx.clip(lower=-1.0, upper=1.0)  # monthly returns guardrail
spx = spx.ffill()

feat = pd.concat([
    fac,
    spx.shift(1).rename(columns={"spx_ret_1m":"spx_ret_lag1"}),
    spx["spx_ret_1m"].rolling(3, min_periods=3).std().shift(1).rename("spx_vol_lag3"),
    spx["spx_ret_1m"].rolling(6, min_periods=6).mean().shift(1).rename("spx_ret_mean6"),
], axis=1)

# ===== Target: next-month direction =====
y = (spx["spx_ret_1m"].shift(-1) > 0).astype(int).rename("spx_dir_fwd1")

Xy = pd.concat([feat, y], axis=1).dropna()
X, y = Xy.drop(columns=["spx_dir_fwd1"]), Xy["spx_dir_fwd1"]

# ===== Time splits =====
split1, split2 = "2016-12-01", "2020-12-01"
X_tr, y_tr = X[X.index <= split1], y[y.index <= split1]
X_va, y_va = X[(X.index > split1) & (X.index <= split2)], y[(y.index > split1) & (y.index <= split2)]
X_te, y_te = X[X.index > split2], y[y.index > split2]

pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(max_iter=1000))
]).fit(X_tr, y_tr)

def cls_metrics(y_true, y_pred):
    return {
        "acc":  float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec":  float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":   float(f1_score(y_true, y_pred, zero_division=0)),
    }

metrics = {
    "train": cls_metrics(y_tr, pipe.predict(X_tr)),
    "valid": cls_metrics(y_va, pipe.predict(X_va)),
    "test":  cls_metrics(y_te, pipe.predict(X_te)),
}

joblib.dump(pipe, ART/"clf_spx_dir_v1.pkl")
joblib.dump(pca,  ART/"cpi_pca_clf_spx_v1.pkl")
with open(ART/"feature_config_clf.json","w") as f:
    json.dump({
        "n_factors": n_factors,
        "features": list(X.columns),
        "splits": [split1, split2],
        "metrics": metrics
    }, f, indent=2)

print("✅ Saved classifier artifacts to", ART)
print("📊 Metrics:", metrics)

