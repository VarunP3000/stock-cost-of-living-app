from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

DATA = Path("data/processed")
ART  = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)

df = (pd.read_csv(DATA/"spx_cpi_global.csv", parse_dates=["ds"])
        .sort_values(["geo","ds"]))

# ---------- Build CPI YoY panel (time x geo) robustly for PCA ----------
panel_raw = (df.pivot_table(index="ds", columns="geo", values="cpi_yoy")
               .sort_index())

# 1) Remove non-finite
panel = panel_raw.replace([np.inf, -np.inf], np.nan)

# 2) (Optional) winsorize extremes per column to tame spikes
q_low  = panel.quantile(0.005)
q_high = panel.quantile(0.995)
panel = panel.clip(lower=q_low, upper=q_high, axis=1)

# 3) Drop super-sparse geos (need enough data)
min_months = 60  # require >= 5 years of monthly data
good_cols = panel.columns[panel.notna().sum() >= min_months]
panel = panel[good_cols]

# 4) Z-score by column (keep NaNs), then fill remaining NaNs with 0 (mean in z-space)
col_means = panel.mean(skipna=True)
col_stds  = panel.std(skipna=True, ddof=0).replace(0, np.nan)
panel_z   = (panel - col_means) / col_stds
panel_z   = panel_z.fillna(0.0)

# 5) Choose a safe number of factors
n_samples, n_features = panel_z.shape
n_factors = min(3, n_samples, n_features) if n_features > 0 else 1

pca = PCA(n_components=n_factors)
factors = pca.fit_transform(panel_z.values)
fac = pd.DataFrame(
    factors, index=panel_z.index,
    columns=[f"cpi_fac{i+1}" for i in range(n_factors)]
)

# ---------- Market context features (lagged to avoid look-ahead) ----------
spx = (df[["ds","spx_ret_1m"]]
         .drop_duplicates("ds")
         .set_index("ds")
         .sort_index())

feat = pd.concat([
    fac,
    spx.shift(1).rename(columns={"spx_ret_1m": "spx_ret_lag1"}),
    spx["spx_ret_1m"].rolling(3).std().shift(1).rename("spx_vol_lag3"),
    spx["spx_ret_1m"].rolling(6).mean().shift(1).rename("spx_ret_mean6"),
], axis=1)

# ---------- Target: next-month SPX return ----------
y = spx["spx_ret_1m"].shift(-1).rename("spx_ret_fwd1")
Xy = pd.concat([feat, y], axis=1).dropna()
X, y = Xy.drop(columns=["spx_ret_fwd1"]), Xy["spx_ret_fwd1"]

# ---------- Time-based split ----------
split1, split2 = "2016-12-01", "2020-12-01"
X_tr, y_tr = X[X.index <= split1], y[y.index <= split1]
X_va, y_va = X[(X.index > split1) & (X.index <= split2)], y[(y.index > split1) & (y.index <= split2)]
X_te, y_te = X[X.index > split2], y[y.index > split2]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=np.logspace(-4, 2, 30)))
]).fit(X_tr, y_tr)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

metrics = {
    "train": {
        "rmse": rmse(y_tr, pipe.predict(X_tr)),
        "r2": float(r2_score(y_tr, pipe.predict(X_tr))),
    },
    "valid": {
        "rmse": rmse(y_va, pipe.predict(X_va)),
        "r2": float(r2_score(y_va, pipe.predict(X_va))),
    },
    "test": {
        "rmse": rmse(y_te, pipe.predict(X_te)),
        "r2": float(r2_score(y_te, pipe.predict(X_te))),
    },
}

# ---------- Save artifacts (outside src/) ----------
joblib.dump(pipe, ART/"ridge_spx_v1.pkl")
joblib.dump(pca,  ART/"cpi_pca_v1.pkl")
with open(ART/"feature_config.json", "w") as f:
    json.dump({
        "n_factors": int(n_factors),
        "features": list(X.columns),
        "splits": [split1, split2],
        "metrics": metrics
    }, f, indent=2)

print("✅ Saved model artifacts to", ART)
print("📊 Metrics:", metrics)
