# src/models/train_quantile.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA = Path("data/processed")
ART  = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)

def build_base_features(n_factors=3):
    df = pd.read_csv(DATA/"spx_cpi_global.csv", parse_dates=["ds"]).sort_values(["geo","ds"])

    # ---- build CPI panel (time x geo) and CLEAN it ----
    panel = (df.pivot_table(index="ds", columns="geo", values="cpi_yoy")
               .sort_index())

    # 1) Ensure numeric
    panel = panel.apply(pd.to_numeric, errors="coerce")

    # 2) Nuke inf/-inf -> NaN
    panel = panel.replace([np.inf, -np.inf], np.nan)

    # 3) Drop degenerate cols/rows (all NaN)
    panel = panel.dropna(how="all", axis=0).dropna(how="all", axis=1)

    # 4) (Optional) Winsorize extreme magnitudes to avoid rogue columns
    #    Clip to the 0.5th–99.5th percentile per column if enough data
    if panel.shape[1] > 0:
        q_low  = panel.quantile(0.005)
        q_high = panel.quantile(0.995)
        panel = panel.clip(lower=q_low, upper=q_high, axis=1)

    # 5) Forward-fill within each column (time continuity), then median-fill leftovers
    panel = panel.ffill().bfill()
    panel = panel.fillna(panel.median())

    # ---- PCA factors ----
    from sklearn.decomposition import PCA
    n_factors = min(n_factors, max(1, panel.shape[1]))
    pca = PCA(n_components=n_factors, random_state=0)
    fac = pd.DataFrame(
        pca.fit_transform(panel),
        index=panel.index,
        columns=[f"cpi_fac{i+1}" for i in range(n_factors)]
    )

    # ---- SPX features ----
    spx = (df[["ds","spx_ret_1m"]].drop_duplicates("ds")
             .set_index("ds").sort_index())
    feat = pd.concat([
        fac,
        spx.shift(1).rename(columns={"spx_ret_1m":"spx_ret_lag1"}),
        spx["spx_ret_1m"].rolling(3).std().shift(1).rename("spx_vol_lag3"),
        spx["spx_ret_1m"].rolling(6).mean().shift(1).rename("spx_ret_mean6"),
    ], axis=1)

    # final cleanup before modeling
    feat = feat.replace([np.inf, -np.inf], np.nan)

    y = spx["spx_ret_1m"].shift(-1).rename("spx_ret_fwd1")
    Xy = pd.concat([feat, y], axis=1).dropna()

    # basic time split
    split1, split2 = "2016-12-01", "2020-12-01"
    X, y = Xy.drop(columns=["spx_ret_fwd1"]), Xy["spx_ret_fwd1"]
    X_tr, y_tr = X[X.index <= split1], y[y.index <= split1]
    X_va, y_va = X[(X.index > split1) & (X.index <= split2)], y[(y.index > split1) & (y.index <= split2)]
    X_te, y_te = X[X.index > split2], y[y.index > split2]
    return (X_tr,y_tr,X_va,y_va,X_te,y_te), pca, list(X.columns), [split1, split2]


def rmse(a,b): return mean_squared_error(a,b)**0.5

def fit_quantile(alpha, X_tr, y_tr):
    # GBM with quantile loss; keep it small & stable
    return GradientBoostingRegressor(
        loss="quantile", alpha=alpha,
        n_estimators=500, learning_rate=0.03, max_depth=3, random_state=0
    ).fit(X_tr, y_tr)

def main():
    (X_tr,y_tr,X_va,y_va,X_te,y_te), pca, feat_names, splits = build_base_features(n_factors=3)

    models = {}
    metrics = {}
    for name, alpha in [("p10",0.10),("p50",0.50),("p90",0.90)]:
        m = fit_quantile(alpha, X_tr, y_tr)
        models[name] = m
        metrics[name] = {
            "train": {"rmse": rmse(y_tr, m.predict(X_tr)), "r2": r2_score(y_tr, m.predict(X_tr))},
            "valid": {"rmse": rmse(y_va, m.predict(X_va)), "r2": r2_score(y_va, m.predict(X_va))},
            "test":  {"rmse": rmse(y_te, m.predict(X_te)), "r2": r2_score(y_te, m.predict(X_te))},
        }
        joblib.dump(m, ART/f"gb_quantile_{name}.pkl")

    # save config (one file for all quantiles)
    with open(ART/"quantile_config.json","w") as f:
        json.dump({
            "quantiles": ["p10","p50","p90"],
            "features": feat_names,
            "splits": splits,
            "metrics": metrics
        }, f, indent=2)

    # also save PCA used for features
    joblib.dump(pca, ART/"cpi_pca_v_quantile.pkl")
    print("✅ Saved quantile models & config to models/artifacts/")
    print("📊 Metrics:", metrics)

if __name__ == "__main__":
    main()
