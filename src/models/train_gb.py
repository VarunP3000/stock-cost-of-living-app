from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import build_features, eval_dict, save_artifacts

def main():
    X_tr,y_tr,X_va,y_va,X_te,y_te,pca,feat_names,splits = build_features(n_factors=3)

    # Simple, strong GB baseline
    gb = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("gb", GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            random_state=42
        ))
    ])

    gb.fit(X_tr, y_tr)

    p_tr = gb.predict(X_tr)
    p_va = gb.predict(X_va)
    p_te = gb.predict(X_te)

    metrics = eval_dict(y_tr, p_tr, y_va, p_va, y_te, p_te)

    # Save
    save_artifacts(gb, pca, feat_names, splits, metrics, prefix="gb_spx_v1")
    print("✅ Saved GradientBoosting artifacts as gb_spx_v1")
    print("📊 Metrics:", metrics)

if __name__ == "__main__":
    main()
