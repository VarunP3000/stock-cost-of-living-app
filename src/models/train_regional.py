# src/models/train_regional.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA = Path("data/processed")
ART  = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)

# Map each geo into a region (edit as you like)
REGIONS = {
    "AMERICAS": {"United States","Canada","Mexico","Brazil","Argentina","Chile","Colombia","Peru"},
    "EMEA":     {"United Kingdom","Germany","France","Italy","Spain","Netherlands","Sweden","Norway","Switzerland","South Africa","Turkey"},
    "APAC":     {"Japan","China","India","South Korea","Australia","New Zealand","Singapore","Hong Kong","Indonesia","Malaysia","Thailand","Philippines","Taiwan","Vietnam"}
}

def build_region_factors(df: pd.DataFrame, geos: list[str], n_factors: int = 2):
    """
    Build PCA factors from CPI YoY for a set of geos (the 'region').
    Returns (fac_df, fitted_pca).
    """
    # --- build time x geo panel for the region ---
    panel = (df[df["geo"].isin(geos)]
                .pivot_table(index="ds", columns="geo", values="cpi_yoy")
                .sort_index())

    # --- CLEANING STEPS (important for PCA) ---
    # 1) force numeric
    panel = panel.apply(pd.to_numeric, errors="coerce")

    # 2) remove inf/-inf
    panel = panel.replace([np.inf, -np.inf], np.nan)

    # 3) drop all-NaN rows/cols
    panel = panel.dropna(how="all", axis=0).dropna(how="all", axis=1)

    if panel.empty or panel.shape[1] == 0:
        raise ValueError(f"Regional panel is empty after cleaning for geos={geos}")

    # 4) tame outliers per column (winsorize via clipping percentiles)
    q_low  = panel.quantile(0.005)
    q_high = panel.quantile(0.995)
    panel = panel.clip(lower=q_low, upper=q_high, axis=1)

    # 5) forward/back fill over time, then median-fill leftovers
    panel = panel.ffill().bfill()
    panel = panel.fillna(panel.median())

    # sanity: no inf remains
    if np.isinf(panel.to_numpy()).any():
        raise ValueError("Found ±inf in regional panel after cleaning")

    # --- PCA ---
    from sklearn.decomposition import PCA
    n_factors = min(n_factors, max(1, panel.shape[1]))
    pca = PCA(n_components=n_factors, random_state=0)
    fac = pd.DataFrame(
        pca.fit_transform(panel),
        index=panel.index,
        columns=[f"reg_fac{i+1}" for i in range(n_factors)]
    )
    return fac, pca

def rmse(a,b): return mean_squared_error(a,b)**0.5

def main():
    df = pd.read_csv(DATA/"spx_cpi_global.csv", parse_dates=["ds"]).sort_values(["geo","ds"])
    spx = (df[["ds","spx_ret_1m"]].drop_duplicates("ds")
             .set_index("ds").sort_index())

    results = {}
    for region, geos in REGIONS.items():
        fac, pca = build_region_factors(df, geos, n_factors=2)
        if fac is None:
            print(f"⚠️ Skipping {region}: no data")
            continue

        feat = pd.concat([
            fac.add_prefix(f"{region.lower()}_"),
            spx.shift(1).rename(columns={"spx_ret_1m":"spx_ret_lag1"}),
        ], axis=1)
        y = spx["spx_ret_1m"].shift(-1).rename("spx_ret_fwd1")
        Xy = pd.concat([feat, y], axis=1).dropna()
        X, y = Xy.drop(columns=["spx_ret_fwd1"]), Xy["spx_ret_fwd1"]

        split1, split2 = "2016-12-01", "2020-12-01"
        X_tr, y_tr = X[X.index <= split1], y[y.index <= split1]
        X_va, y_va = X[(X.index > split1) & (X.index <= split2)], y[(y.index > split1) & (y.index <= split2)]
        X_te, y_te = X[X.index > split2], y[y.index > split2]

        pipe = Pipeline([("scaler", StandardScaler()),
                         ("ridge", RidgeCV(alphas=np.logspace(-4,2,30)))])
        pipe.fit(X_tr, y_tr)

        metrics = {
            "train": {"rmse": rmse(y_tr, pipe.predict(X_tr)), "r2": r2_score(y_tr, pipe.predict(X_tr))},
            "valid": {"rmse": rmse(y_va, pipe.predict(X_va)), "r2": r2_score(y_va, pipe.predict(X_va))},
            "test":  {"rmse": rmse(y_te, pipe.predict(X_te)), "r2": r2_score(y_te, pipe.predict(X_te))},
        }

        # save per-region artifacts
        joblib.dump(pipe, ART/f"ridge_spx_{region.lower()}.pkl")
        joblib.dump(pca,  ART/f"cpi_pca_{region.lower()}.pkl")
        with open(ART/f"regional_{region.lower()}_config.json","w") as f:
            json.dump({
                "region": region,
                "geos": sorted(list(geos)),
                "features": list(X.columns),
                "splits": [split1, split2],
                "metrics": metrics
            }, f, indent=2)

        results[region] = metrics

    with open(ART/"regional_index.json","w") as f:
        json.dump({"regions": sorted(list(REGIONS.keys())), "results": results}, f, indent=2)

    print("✅ Saved regional models to models/artifacts/")
    print("📊 Metrics:", results)

if __name__ == "__main__":
    main()
