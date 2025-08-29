from pathlib import Path
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import build_features, eval_dict, save_artifacts

def main():
    X_tr,y_tr,X_va,y_va,X_te,y_te,pca,feat_names,splits = build_features(n_factors=3)

    # ElasticNet with time-series aware alphas (logspace) and multiple l1 ratios
    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 1.5, 40),
            cv=5,
            n_jobs=None,
            random_state=42
        ))
    ])

    enet.fit(X_tr, y_tr)

    p_tr = enet.predict(X_tr)
    p_va = enet.predict(X_va)
    p_te = enet.predict(X_te)

    metrics = eval_dict(y_tr, p_tr, y_va, p_va, y_te, p_te)

    # Save
    save_artifacts(enet, pca, feat_names, splits, metrics, prefix="enet_spx_v1")
    print("✅ Saved ElasticNet artifacts as enet_spx_v1")
    print("📊 Metrics:", metrics)

if __name__ == "__main__":
    main()
