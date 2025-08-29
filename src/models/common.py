from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

DATA = Path("data/processed")
ART  = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)

def _rmse(y_true, y_pred):
    # use squared=True for maximum compatibility, then sqrt
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _sanitize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    # replace +/-inf with NaN then fill
    panel = panel.replace([np.inf, -np.inf], np.nan)

    # forward-fill within each column (monthly data should be OK)
    panel = panel.ffill()

    # robust clip: cap extreme z-scores to avoid blowing up PCA
    # compute per-column z, ignoring NaNs
    def _clip_col(s: pd.Series, z=6.0):
        m, sd = s.mean(skipna=True), s.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            return s
        return s.clip(lower=m - z*sd, upper=m + z*sd)
    panel = panel.apply(_clip_col)

    # final fill with column medians
    panel = panel.fillna(panel.median(numeric_only=True))

    # drop any columns that are still entirely NA (shouldn’t happen)
    panel = panel.dropna(axis=1, how="all")
    return panel

def build_features(n_factors=3,
                   split1="2016-12-01", split2="2020-12-01"):
    """
    Loads panel, builds CPI PCA factors + market lags, returns time-split matrices.
    Returns:
      X_tr,y_tr,X_va,y_va,X_te,y_te,pca,feat_names,splits(dict)
    """
    df = pd.read_csv(DATA/"spx_cpi_global.csv", parse_dates=["ds"]).sort_values(["geo","ds"])

    # Pivot CPI YoY to time×geo
    panel = (df.pivot_table(index="ds", columns="geo", values="cpi_yoy")
               .sort_index())
    panel = _sanitize_panel(panel)

    # PCA factors (global CPI structure)
    pca = PCA(n_components=n_factors, random_state=42)
    fac = pd.DataFrame(
        pca.fit_transform(panel),
        index=panel.index,
        columns=[f"cpi_fac{i+1}" for i in range(n_factors)]
    )

    # Market context (same as baseline Ridge)
    spx = (df[["ds","spx_ret_1m"]].drop_duplicates("ds")
           .set_index("ds").sort_index())

    feat = pd.concat([
        fac,
        spx.shift(1).rename(columns={"spx_ret_1m":"spx_ret_lag1"}),
        spx["spx_ret_1m"].rolling(3).std().shift(1).rename("spx_vol_lag3"),
        spx["spx_ret_1m"].rolling(6).mean().shift(1).rename("spx_ret_mean6"),
    ], axis=1)

    # target next-month return
    y = spx["spx_ret_1m"].shift(-1).rename("spx_ret_fwd1")
    Xy = pd.concat([feat, y], axis=1).dropna()
    X, y = Xy.drop(columns=["spx_ret_fwd1"]), Xy["spx_ret_fwd1"]
    feat_names = list(X.columns)

    # time splits
    X_tr, y_tr = X[X.index <= split1], y[y.index <= split1]
    X_va, y_va = X[(X.index > split1) & (X.index <= split2)], y[(y.index > split1) & (y.index <= split2)]
    X_te, y_te = X[X.index > split2], y[y.index > split2]

    splits = {"split1": split1, "split2": split2}
    return X_tr,y_tr,X_va,y_va,X_te,y_te,pca,feat_names,splits

def eval_dict(y_tr, p_tr, y_va, p_va, y_te, p_te):
    return {
        "train": {"rmse": _rmse(y_tr, p_tr), "r2": float(r2_score(y_tr, p_tr))},
        "valid": {"rmse": _rmse(y_va, p_va), "r2": float(r2_score(y_va, p_va))},
        "test":  {"rmse": _rmse(y_te, p_te), "r2": float(r2_score(y_te, p_te))},
    }

def save_artifacts(model, pca, feature_names, splits, metrics, prefix: str):
    import joblib, json
    joblib.dump(model, ART/f"{prefix}.pkl")
    joblib.dump(pca,   ART/f"cpi_pca_{prefix}.pkl")
    with open(ART/f"feature_config_{prefix}.json","w") as f:
        json.dump({
            "n_factors": len([c for c in feature_names if c.startswith("cpi_fac")]),
            "features": feature_names,
            "splits": [splits["split1"], splits["split2"]],
            "metrics": metrics
        }, f, indent=2)
