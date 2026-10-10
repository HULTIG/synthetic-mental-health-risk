"""Shared modelling, evaluation and explanation utilities."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from generator import FEATURES, NOMINAL

warnings.filterwarnings("ignore")
NUMERIC = [f for f in FEATURES if f not in NOMINAL]


def frame(X: dict) -> pd.DataFrame:
    return pd.DataFrame({f: X[f] for f in FEATURES})


def split(X: dict, target: str, seed: int, test_size=0.2):
    y = X["phq_class"] if target == "phq" else X["gad_class"]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return tr, te, y


def subset(X: dict, idx) -> dict:
    return {k: v[idx] for k, v in X.items()}


# ----------------------------------------------------------------------------- models
def linear_pre():
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), NOMINAL)])


class OrdinalProbit:
    """Cumulative-link (probit) model on one-hot / standardised features."""

    def __init__(self):
        self.pre = linear_pre()

    def fit(self, X, y):
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        Z = self.pre.fit_transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        self.k = int(y.max()) + 1
        self.res = OrderedModel(y, Z, distr="probit").fit(method="bfgs", disp=False, maxiter=2000)
        self.classes_ = np.arange(self.k)
        return self

    def predict_proba(self, X):
        Z = self.pre.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        return np.asarray(self.res.model.predict(self.res.params, exog=Z))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def make_model(name: str, params: dict | None = None, seed: int = 0):
    p = dict(params or {})
    if name == "LR":
        return Pipeline([("pre", linear_pre()), ("clf", LogisticRegression(max_iter=3000, C=p.get("C", 1.0)))])
    if name == "OrdinalProbit":
        return OrdinalProbit()
    if name == "SVM":
        return Pipeline([("pre", linear_pre()), ("clf", SVC(probability=True, random_state=seed,
                                                           C=p.get("C", 1.0), gamma=p.get("gamma", "scale")))])
    if name == "MLP":
        return Pipeline([("pre", linear_pre()), ("clf", MLPClassifier(
            hidden_layer_sizes=p.get("hidden", (64, 32)), alpha=p.get("alpha", 1e-3),
            learning_rate_init=p.get("lr", 1e-3), max_iter=600, early_stopping=True, random_state=seed))])
    if name == "RF":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth"),
                                      min_samples_leaf=p.get("min_samples_leaf", 1),
                                      max_features=p.get("max_features", "sqrt"), n_jobs=2, random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=2, random_state=seed,
                             eval_metric="mlogloss", verbosity=0, **p)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=2, verbose=-1, **p)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=2,
                                  loss_function="MultiClass", **p)
    raise ValueError(name)


# ----------------------------------------------------------------------------- metrics
def scores(y, proba, k):
    proba = np.clip(proba, 1e-9, 1)
    proba = proba / proba.sum(1, keepdims=True)
    pred = proba.argmax(1)
    return {"macro_f1": f1_score(y, pred, average="macro", labels=list(range(k)), zero_division=0),
            "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, proba, labels=list(range(k)))}


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


# ----------------------------------------------------------------------------- explanations
def expected_class(proba):
    return proba @ np.arange(proba.shape[1])


def pd_curve(predict_proba, Xdf: pd.DataFrame, feat: str, grid):
    out = []
    for v in grid:
        Z = Xdf.copy()
        Z[feat] = v
        out.append(expected_class(predict_proba(Z)).mean())
    return np.array(out)


def ale_curve(predict_proba, Xdf: pd.DataFrame, feat: str, edges):
    """First-order ALE of the expected class index; returns values at bin edges."""
    x = Xdf[feat].values
    bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        m = bins == b
        if not m.any():
            continue
        lo, hi = Xdf[m].copy(), Xdf[m].copy()
        lo[feat] = edges[b]
        hi[feat] = edges[b + 1]
        effects[b] = (expected_class(predict_proba(hi)) - expected_class(predict_proba(lo))).mean()
        counts[b] = m.sum()
    acc = np.concatenate([[0], np.cumsum(effects)])
    # centre with respect to the data distribution
    mid = (acc[:-1] + acc[1:]) / 2
    return acc - np.sum(mid * counts) / counts.sum()


def centre(curve, weights=None):
    w = np.ones_like(curve) if weights is None else weights
    return curve - np.sum(curve * w) / np.sum(w)


def rank_recovery(attr: dict, truth: dict):
    f = FEATURES
    a = np.array([attr[k] for k in f])
    t = np.array([truth[k] for k in f])
    tau = stats.kendalltau(a, t).statistic
    rho = stats.spearmanr(a, t).statistic
    proxy = np.array([truth[k] == 0 for k in f])
    share = a[proxy].sum() / a.sum() if a.sum() > 0 else np.nan
    top_true = set(np.array(f)[np.argsort(-t)[:3]])
    top_attr = set(np.array(f)[np.argsort(-a)[:3]])
    return {"kendall": tau, "spearman": rho, "proxy_share": share,
            "top3_hit": len(top_true & top_attr) / 3}
