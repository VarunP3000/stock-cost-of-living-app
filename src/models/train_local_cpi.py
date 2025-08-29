from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA = Path("data/processed")
ART  = Path("models/artifacts/local_cpi"); ART.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA/"spx_cpi_global.csv", parse_dates=["ds"]).sort_values(["geo","ds"])

geos = df["geo"].dropna().unique().tolist()
results = {}

def rmse(y_true, y_pred): 
    # SciKit < 0.22 compat: compute RMSE manually
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

for geo in geos:
    sub = df[df["geo"]==geo].set_index("ds").sort_index()

    # Clean series: remove inf, clip outliers, ffill
    sub["cpi_yoy"] = sub["cpi_yoy"].replace([np.inf, -np.inf], np.nan)
    sub["cpi_yoy"] = sub["cpi_yoy"].clip(lower=-100, upper=100)
    sub["cpi_yoy"] = sub["cpi_yoy"].ffill()

    sub["cpi_yoy_fwd1"] = sub["cpi_yoy"].shift(-1)

    Xy = sub[["cpi_yoy","cpi_yoy_fwd1"]].dropna()
    if len(Xy) < 60:
        continue  # need enough history

    X, y = Xy[["cpi_yoy"]], Xy["cpi_yoy_fwd1"]

    split1, split2 = "2016-12-01", "2020-12-01"
    X_tr, y_tr = X[X.index <= split1], y[y.index <= split1]
    X_va, y_va = X[(X.index > split1) & (X.index <= split2)], y[(y.index > split1) & (y.index <= split2)]
    X_te, y_te = X[X.index > split2], y[y.index > split2]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 2, 30)))
    ]).fit(X_tr, y_tr)

    m = {
        "train": {"rmse": rmse(y_tr, pipe.predict(X_tr)), "r2": float(r2_score(y_tr, pipe.predict(X_tr)))},
        "valid": {"rmse": rmse(y_va, pipe.predict(X_va)), "r2": float(r2_score(y_va, pipe.predict(X_va)))},
        "test":  {"rmse": rmse(y_te, pipe.predict(X_te)), "r2": float(r2_score(y_te, pipe.predict(X_te)))},
    }

    results[geo] = m
    joblib.dump(pipe, ART/f"{geo}_cpi_model.pkl")

with open(ART/"metrics.json","w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Trained local CPI regressors for {len(results)} geos, saved to {ART}")
