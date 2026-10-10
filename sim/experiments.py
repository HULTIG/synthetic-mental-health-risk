"""All experiments of the study. Usage: python sim/experiments.py <stage> [stage ...]

Stages
  bench     predictive benchmark at the reference configuration (10 seeds, 8 learners + Bayes)
  signal    signal-strength sweep (R^2), XGBoost and ordinal probit against the Bayes ceiling
  xai       explanation recovery over dependence x signal (TreeSHAP, permutation, PDP, ALE)
  kernel    KernelSHAP at the reference configuration
  ablation  redundant PA encodings removed vs kept
  size      sample-size sweep
  coupling  PHQ-GAD correlation under cross-term and correlated-noise couplings
"""
import json, sys, time, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, "sim")
from generator import Config, Generator, FEATURES
from common import (frame, split, subset, make_model, scores, pd_curve, ale_curve, centre,
                    rank_recovery, expected_class)

HP = json.load(open("results/hyperparams.json"))
REF = dict(lam=1.0, r2=0.8)
K = {"phq": 5, "gad": 4}
GEN_CACHE = {}


def gen(lam, r2, **kw):
    key = (lam, r2, tuple(sorted(kw.items())))
    if key not in GEN_CACHE:
        GEN_CACHE[key] = Generator(Config(lam=lam, r2_phq=r2, r2_gad=r2, **kw))
    return GEN_CACHE[key]


def params(target, name):
    p = HP[target][name]["params"]
    if name == "MLP" and "hidden" in p:
        p = dict(p, hidden=tuple(p["hidden"]))
    return p


def as_dict(df):
    return {c: df[c].values for c in df.columns}


def true_proba_fn(g, target):
    return lambda df: g.class_proba(as_dict(df), target)


def save(rows, name):
    pd.DataFrame(rows).to_csv(f"results/{name}.csv", index=False)


# ------------------------------------------------------------------------- benchmark
def stage_bench(seeds=range(10)):
    rows = []
    g = gen(**REF)
    for seed in seeds:
        X = g.sample(seed=seed)
        df = frame(X)
        for target in ["phq", "gad"]:
            tr, te, y = split(X, target, seed)
            P = g.class_proba(subset(X, te), target)
            rows.append(dict(seed=seed, target=target, model="Bayes", **scores(y[te], P, K[target])))
            for name in ["LR", "OrdinalProbit", "SVM", "MLP", "RF", "XGBoost", "LightGBM", "CatBoost"]:
                t0 = time.time()
                m = make_model(name, params(target, name) if name in HP[target] else None, seed)
                m.fit(df.iloc[tr], y[tr])
                s = scores(y[te], m.predict_proba(df.iloc[te]), K[target])
                rows.append(dict(seed=seed, target=target, model=name, fit_s=time.time() - t0, **s))
            save(rows, "bench")
            print("bench", seed, target, flush=True)


# ------------------------------------------------------------------------- signal sweep
def stage_signal(seeds=range(5)):
    rows = []
    for r2 in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
        g = gen(1.0, r2)
        for seed in seeds:
            X = g.sample(seed=seed)
            df = frame(X)
            for target in ["phq", "gad"]:
                tr, te, y = split(X, target, seed)
                P = g.class_proba(subset(X, te), target)
                rows.append(dict(r2=r2, seed=seed, target=target, model="Bayes", **scores(y[te], P, K[target])))
                for name in ["OrdinalProbit", "XGBoost"]:
                    m = make_model(name, params(target, name) if name in HP[target] else None, seed)
                    m.fit(df.iloc[tr], y[tr])
                    rows.append(dict(r2=r2, seed=seed, target=target, model=name,
                                     **scores(y[te], m.predict_proba(df.iloc[te]), K[target])))
            save(rows, "signal")
        print("signal", r2, flush=True)


# ------------------------------------------------------------------------- explanations
def tree_shap(model, bg, Xe, kind):
    import shap
    if kind == "interventional":
        ex = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="raw")
    else:
        ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = ex.shap_values(Xe, check_additivity=False)
    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)  # (n, p, k)
    return np.abs(sv).mean(0).sum(-1), sv


def perm_importance(model, Xe, y, k, rng, repeats=5):
    from sklearn.metrics import log_loss
    base = log_loss(y, model.predict_proba(Xe), labels=list(range(k)))
    imp = {}
    for f in Xe.columns:
        d = []
        for _ in range(repeats):
            Z = Xe.copy()
            Z[f] = rng.permutation(Z[f].values)
            d.append(log_loss(y, model.predict_proba(Z), labels=list(range(k))) - base)
        imp[f] = max(np.mean(d), 0.0)
    return imp


def explain_run(g, X, target, seed, feats, n_explain=300, n_bg=50, curves=True, store_curves=False):
    df = frame(X)[feats]
    tr, te, y = split(X, target, seed)
    model = make_model("XGBoost", params(target, "XGBoost"), seed).fit(df.iloc[tr], y[tr])
    rng = np.random.default_rng(seed)
    Xte = df.iloc[te]
    perf = scores(y[te], model.predict_proba(Xte), K[target])
    Xe = Xte.iloc[rng.choice(len(te), min(n_explain, len(te)), replace=False)]
    bg = df.iloc[tr].iloc[rng.choice(len(tr), n_bg, replace=False)]
    truth = {f: v for f, v in g.true_importance(target).items() if f in feats}
    out = dict(perf=perf, methods={})
    for kind in ["interventional", "path_dependent"]:
        imp, _ = tree_shap(model, bg, Xe, kind)
        out["methods"][f"TreeSHAP-{kind}"] = dict(zip(feats, imp))
    out["methods"]["Permutation"] = perm_importance(model, Xte, y[te], K[target], rng)
    out["recovery"] = {m: rank_recovery_sub(a, truth, feats) for m, a in out["methods"].items()}
    if curves:
        full = frame(X)  # true model needs every column
        Xte_full = full.iloc[te]
        tp = true_proba_fn(g, target)
        out["curves"] = {}
        for feat in ["pa_minutes_week", "sleep_hours", "age"]:
            lo, hi = np.quantile(Xte[feat], [0.05, 0.95])
            grid = np.linspace(lo, hi, 15)
            edges = np.unique(np.quantile(Xte[feat], np.linspace(0.05, 0.95, 11)))
            mp = lambda Z: model.predict_proba(Z[feats])
            pd_m = centre(pd_curve(mp, Xte_full, feat, grid))
            pd_t = centre(pd_curve(tp, Xte_full, feat, grid))
            ale_m = ale_curve(mp, Xte_full, feat, edges)
            ale_t = ale_curve(tp, Xte_full, feat, edges)
            pd_t_at_edges = centre(pd_curve(tp, Xte_full, feat, edges))
            rec = dict(pd_rmse=float(np.sqrt(np.mean((pd_m - pd_t) ** 2))),
                       ale_rmse_vs_truePD=float(np.sqrt(np.mean((ale_m - pd_t_at_edges) ** 2))),
                       ale_rmse_vs_trueALE=float(np.sqrt(np.mean((ale_m - ale_t) ** 2))),
                       pd_range_model=float(np.ptp(pd_m)), ale_range_model=float(np.ptp(ale_m)),
                       range_true=float(np.ptp(pd_t)))
            if store_curves:
                rec.update(grid=grid.tolist(), pd_model=pd_m.tolist(), pd_true=pd_t.tolist(),
                           edges=edges.tolist(), ale_model=ale_m.tolist(), ale_true=ale_t.tolist(),
                           pd_true_edges=pd_t_at_edges.tolist())
            out["curves"][feat] = rec
    return out


def rank_recovery_sub(attr, truth, feats):
    if len(feats) == len(FEATURES):
        return rank_recovery(attr, truth)
    # reduced feature set: compute metrics on the available features only
    from scipy import stats
    a = np.array([attr[k] for k in feats]); t = np.array([truth[k] for k in feats])
    proxy = t == 0
    return {"kendall": stats.kendalltau(a, t).statistic, "spearman": stats.spearmanr(a, t).statistic,
            "proxy_share": a[proxy].sum() / a.sum(),
            "top3_hit": len(set(np.argsort(-t)[:3]) & set(np.argsort(-a)[:3])) / 3}


def flatten(meta, out):
    rows = []
    for m, rec in out["recovery"].items():
        r = dict(meta, method=m, **rec, **{f"perf_{k}": v for k, v in out["perf"].items()})
        tot = sum(out["methods"][m].values())
        r.update({f"share_{f}": v / tot for f, v in out["methods"][m].items()})
        rows.append(r)
    return rows


def stage_xai(seeds=range(5)):
    rows, crows, store = [], [], {}
    for lam, r2 in itertools.product([0.0, 0.5, 1.0], [0.4, 0.8, 0.95]):
        g = gen(lam, r2)
        for seed in seeds:
            X = g.sample(seed=seed)
            targets = ["phq", "gad"] if (lam, r2) == (1.0, 0.8) else ["phq"]
            for target in targets:
                ref = (lam, r2) == (1.0, 0.8)
                out = explain_run(g, X, target, seed, FEATURES, store_curves=ref)
                meta = dict(lam=lam, r2=r2, seed=seed, target=target)
                rows += flatten(meta, out)
                for feat, rec in out["curves"].items():
                    crows.append(dict(meta, feature=feat, **{k: v for k, v in rec.items() if not isinstance(v, list)}))
                    if ref:
                        store[f"{target}|{seed}|{feat}"] = rec
            save(rows, "xai"); save(crows, "curves")
            json.dump(store, open("results/curves_reference.json", "w"))
        print("xai", lam, r2, flush=True)


def stage_kernel(seeds=range(3)):
    import shap
    rows = []
    g = gen(**REF)
    for seed in seeds:
        X = g.sample(seed=seed)
        df = frame(X)
        for target in ["phq", "gad"]:
            tr, te, y = split(X, target, seed)
            model = make_model("XGBoost", params(target, "XGBoost"), seed).fit(df.iloc[tr], y[tr])
            rng = np.random.default_rng(seed)
            Xe = df.iloc[te].iloc[rng.choice(len(te), 100, replace=False)]
            bg = shap.sample(df.iloc[tr], 50, random_state=seed)
            t0 = time.time()
            f = lambda Z, m=model: m.predict_proba(pd.DataFrame(Z, columns=FEATURES))
            ex = shap.KernelExplainer(f, np.asarray(bg, dtype=float))
            sv = ex.shap_values(np.asarray(Xe, dtype=float), nsamples=1000, silent=True)
            sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
            imp = dict(zip(FEATURES, np.abs(sv).mean(0).sum(-1)))
            rec = rank_recovery(imp, g.true_importance(target))
            tot = sum(imp.values())
            rows.append(dict(seed=seed, target=target, method="KernelSHAP", secs=time.time() - t0, **rec,
                             **{f"share_{f}": v / tot for f, v in imp.items()}))
            save(rows, "kernel")
            print("kernel", seed, target, flush=True)


def stage_ablation(seeds=range(5)):
    rows, crows = [], []
    g = gen(**REF)
    reduced = [f for f in FEATURES if f not in ("pa_days_week", "pa_minutes_session", "pa_type", "pa_intensity")]
    for seed in seeds:
        X = g.sample(seed=seed)
        for label, feats in [("full", FEATURES), ("non-redundant", reduced)]:
            out = explain_run(g, X, "phq", seed, feats)
            meta = dict(features=label, seed=seed, target="phq")
            rows += flatten(meta, out)
            for feat, rec in out["curves"].items():
                crows.append(dict(meta, feature=feat, **rec))
        save(rows, "ablation"); save(crows, "ablation_curves")
        print("ablation", seed, flush=True)


def stage_size(seeds=range(5)):
    rows = []
    g = gen(**REF)
    for n in [1000, 2000, 3915, 8000, 16000]:
        for seed in seeds:
            X = g.sample(n=n, seed=seed)
            df = frame(X)
            tr, te, y = split(X, "phq", seed)
            P = g.class_proba(subset(X, te), "phq")
            rows.append(dict(n=n, seed=seed, model="Bayes", **scores(y[te], P, 5)))
            for name in ["OrdinalProbit", "XGBoost"]:
                m = make_model(name, params("phq", name) if name in HP["phq"] else None, seed).fit(df.iloc[tr], y[tr])
                rows.append(dict(n=n, seed=seed, model=name, **scores(y[te], m.predict_proba(df.iloc[te]), 5)))
            out = explain_run(g, X, "phq", seed, FEATURES, curves=True)
            for meth, rec in out["recovery"].items():
                rows.append(dict(n=n, seed=seed, model=f"XAI:{meth}", **rec))
            for feat, rec in out["curves"].items():
                rows.append(dict(n=n, seed=seed, model=f"CURVE:{feat}", **rec))
            save(rows, "size")
        print("size", n, flush=True)


def stage_coupling():
    from scipy.stats import spearmanr
    rows = []
    for r2 in [0.4, 0.8, 0.95]:
        for kind, vals in [("independent", [0.0]), ("correlated noise", [0.3, 0.6, 0.9]),
                           ("cross-term", [0.3, 0.6, 0.9])]:
            for v in vals:
                kw = {"noise_corr": v} if kind == "correlated noise" else ({"gad_on_phq": v} if kind == "cross-term" else {})
                g = gen(1.0, r2, **kw)
                X = g.sample(n=100_000, seed=7)
                rows.append(dict(r2=r2, coupling=kind, value=v,
                                 spearman=spearmanr(X["phq9"], X["gad7"]).statistic,
                                 pearson=np.corrcoef(X["phq9"], X["gad7"])[0, 1]))
        save(rows, "coupling")
        print("coupling", r2, flush=True)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        globals()[f"stage_{st}"]()
        print(f"== {st} done in {time.time() - t0:.0f}s", flush=True)
