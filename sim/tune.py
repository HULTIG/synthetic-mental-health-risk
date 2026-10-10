"""Hyper-parameter search on the development split of the reference configuration (seed 0).

The selected settings are then frozen for every other seed and configuration, so the
test data of the experiments never take part in model selection.
"""
import json, sys, time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform, randint, uniform

sys.path.insert(0, "sim")
from generator import Config, Generator
from common import frame, split, make_model

SPACES = {
    "LR": {"clf__C": loguniform(1e-2, 1e2)},
    "SVM": {"clf__C": loguniform(1e-1, 1e2), "clf__gamma": loguniform(1e-3, 1e0)},
    "MLP": {"clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "clf__alpha": loguniform(1e-5, 1e-1), "clf__learning_rate_init": loguniform(1e-4, 1e-2)},
    "RF": {"n_estimators": [300, 500], "max_depth": [None, 6, 10, 16], "min_samples_leaf": randint(1, 20),
           "max_features": ["sqrt", 0.5, 0.8]},
    "XGBoost": {"n_estimators": randint(100, 600), "max_depth": randint(2, 7), "learning_rate": loguniform(0.01, 0.2),
                "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                "min_child_weight": loguniform(1, 20), "reg_lambda": loguniform(0.1, 10)},
    "LightGBM": {"n_estimators": randint(100, 600), "num_leaves": randint(4, 32), "learning_rate": loguniform(0.01, 0.2),
                 "subsample": uniform(0.6, 0.4), "subsample_freq": [1], "colsample_bytree": uniform(0.5, 0.5),
                 "min_child_samples": randint(10, 60), "reg_lambda": loguniform(0.1, 10)},
    "CatBoost": {"iterations": randint(200, 700), "depth": randint(3, 8), "learning_rate": loguniform(0.02, 0.2),
                 "l2_leaf_reg": loguniform(1, 10)},
}
N_ITER = {"LR": 10, "SVM": 12, "MLP": 12, "RF": 12, "XGBoost": 25, "LightGBM": 25, "CatBoost": 12}


def unwrap(name, params):
    out = {}
    for k, v in params.items():
        k = k.replace("clf__", "")
        if name == "MLP":
            k = {"hidden_layer_sizes": "hidden", "learning_rate_init": "lr"}.get(k, k)
        if isinstance(v, (np.integer,)):
            v = int(v)
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out


def main():
    g = Generator(Config(lam=1.0, r2_phq=0.8, r2_gad=0.8))
    X = g.sample(seed=0)
    df = frame(X)
    best = {}
    for target in ["phq", "gad"]:
        tr, te, y = split(X, target, seed=0)
        best[target] = {}
        for name, space in SPACES.items():
            t0 = time.time()
            est = make_model(name, seed=0)
            search = RandomizedSearchCV(est, space, n_iter=N_ITER[name], scoring="f1_macro",
                                        cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                        random_state=0, n_jobs=1, refit=False)
            search.fit(df.iloc[tr], y[tr])
            best[target][name] = {"params": unwrap(name, search.best_params_), "cv_f1": float(search.best_score_)}
            print(target, name, round(search.best_score_, 4), best[target][name]["params"],
                  f"{time.time() - t0:.0f}s", flush=True)
            json.dump(best, open("results/hyperparams.json", "w"), indent=1)


if __name__ == "__main__":
    main()
