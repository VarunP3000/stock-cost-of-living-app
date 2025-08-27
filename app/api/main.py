from pathlib import Path
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd, numpy as np, joblib, json

DATA = Path("data/processed"); ART = Path("models/artifacts")
DF = pd.read_csv(DATA / "spx_cpi_global.csv", parse_dates=["ds"]).sort_values(["geo","ds"])
pipe = joblib.load(ART / "ridge_spx_v1.pkl")
pca  = joblib.load(ART / "cpi_pca_v1.pkl")
with open(ART / "feature_config.json") as f: cfg = json.load(f)

app = FastAPI(title="Inflation→SPX API")

def build_features(asof: str | None = None) -> pd.Series:
    # --- CPI YoY panel prep (mirror training) ---
    panel = DF.pivot_table(index="ds", columns="geo", values="cpi_yoy").sort_index()
    if asof: panel = panel.loc[:asof]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    # winsorize
    q_low, q_high = panel.quantile(0.005), panel.quantile(0.995)
    panel = panel.clip(lower=q_low, upper=q_high, axis=1)
    # min history
    good = panel.columns[panel.notna().sum() >= 60]
    panel = panel[good]
    # z-score, fillna
    means, stds = panel.mean(skipna=True), panel.std(skipna=True, ddof=0).replace(0, np.nan)
    panel_z = ((panel - means) / stds).fillna(0.0)
    # PCA factors
    fac = pd.DataFrame(pca.transform(panel_z.values), index=panel_z.index,
                       columns=[f"cpi_fac{i+1}" for i in range(pca.n_components_)])
    # --- market context (lagged) ---
    spx = (DF[["ds","spx_ret_1m"]].drop_duplicates("ds").set_index("ds").sort_index())
    if asof: spx = spx.loc[:asof]
    feat = pd.concat([
        fac,
        spx.shift(1).rename(columns={"spx_ret_1m":"spx_ret_lag1"}),
        spx["spx_ret_1m"].rolling(3).std().shift(1).rename("spx_vol_lag3"),
        spx["spx_ret_1m"].rolling(6).mean().shift(1).rename("spx_ret_mean6"),
    ], axis=1).dropna()
    # align to training feature list
    X = feat[cfg["features"]].iloc[-1]
    return X

class ForecastOut(BaseModel):
    asof: str
    pred_spx_ret_next_month: float

@app.get("/forecast", response_model=ForecastOut)
def forecast(asof: str | None = Query(default=None)):
    x = build_features(asof)
    pred = float(pipe.predict(x.to_frame().T)[0])
    return {"asof": str(x.name.date()), "pred_spx_ret_next_month": pred}

@app.get("/metrics")
def metrics():
    return cfg["metrics"]
