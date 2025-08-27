from __future__ import annotations
import pandas as pd

def standardize(df: pd.DataFrame, *, ds: str, geo: str, metric: str, value: str) -> pd.DataFrame:
    out = df.rename(columns={ds:"ds", geo:"geo", metric:"metric", value:"value"})[["ds","geo","metric","value"]].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["geo"] = out["geo"].astype(str)
    out["metric"] = out["metric"].astype(str)
    out["value"] = out["value"].astype(float)
    return out.sort_values(["geo","metric","ds"]).reset_index(drop=True)
