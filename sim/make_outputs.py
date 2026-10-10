"""Builds every table, figure and in-text number of the manuscript from results/*.csv."""
import json, sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, "sim")
from plotstyle import plt, INK, INK2, MUTED, BLUE, ORANGE, AQUA, VIOLET, FEATURE_LABELS
from generator import Config, Generator, FEATURES
from common import ci95

os.makedirs("paper/tables", exist_ok=True)
NUM = {}  # macros for numbers.tex


def macro(name, value):
    NUM[name] = value


def fmt(m, s, d=3):
    return f"{m:.{d}f} ({s:.{d}f})"


G = Generator(Config(lam=1.0, r2_phq=0.8, r2_gad=0.8))
TRUTH = {t: G.true_importance(t) for t in ["phq", "gad"]}
METHOD_LABEL = {"TreeSHAP-interventional": "TreeSHAP, interventional",
                "TreeSHAP-path_dependent": "TreeSHAP, path-dependent",
                "KernelSHAP": "KernelSHAP", "Permutation": "Permutation importance"}

# ----------------------------------------------------------------------------- E1 benchmark
def bench():
    d = pd.read_csv("results/bench.csv")
    order = ["Bayes", "OrdinalProbit", "LR", "CatBoost", "LightGBM", "XGBoost", "RF", "MLP", "SVM"]
    names = {"Bayes": "Bayes classifier (ceiling)", "OrdinalProbit": "Ordinal probit", "LR": "Logistic regression",
             "CatBoost": "CatBoost", "LightGBM": "LightGBM", "XGBoost": "XGBoost", "RF": "Random forest",
             "MLP": "Multilayer perceptron", "SVM": "SVM (RBF)"}
    g = d.groupby(["target", "model"])
    lines = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{PHQ-9 (5 classes)} & \\multicolumn{3}{c}{GAD-7 (4 classes)} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
             "Model & Macro-F1 & Accuracy & Log-loss & Macro-F1 & Accuracy & Log-loss \\\\", "\\midrule"]
    best = {}
    for t in ["phq", "gad"]:
        sub = d[(d.target == t) & (d.model != "Bayes")].groupby("model").macro_f1.mean()
        best[t] = sub.idxmax()
    for m in order:
        cells = []
        for t in ["phq", "gad"]:
            q = g.get_group((t, m))
            for met in ["macro_f1", "accuracy", "log_loss"]:
                cells.append(fmt(q[met].mean(), q[met].std()))
        row = names[m] + " & " + " & ".join(cells) + " \\\\"
        lines.append(row)
        if m == "Bayes":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open("paper/tables/tab_bench.tex", "w").write("\n".join(lines) + "\n")
    # numbers
    for t in ["phq", "gad"]:
        T = t.upper()
        bay = g.get_group((t, "Bayes"))
        for m in ["XGBoost", "OrdinalProbit", "CatBoost", "LightGBM", "LR", "RF", "MLP", "SVM"]:
            q = g.get_group((t, m))
            macro(f"{T}{m}F", f"{q.macro_f1.mean():.3f}")
            macro(f"{T}{m}Acc", f"{q.accuracy.mean():.3f}")
            macro(f"{T}{m}LL", f"{q.log_loss.mean():.3f}")
        macro(f"{T}BayesF", f"{bay.macro_f1.mean():.3f}")
        macro(f"{T}BayesAcc", f"{bay.accuracy.mean():.3f}")
        macro(f"{T}BayesLL", f"{bay.log_loss.mean():.3f}")
        # paired differences XGBoost vs probit / Bayes (accuracy, macro-F1)
        piv = d[d.target == t].pivot(index="seed", columns="model", values="accuracy")
        diff = piv["XGBoost"] - piv["Bayes"]
        lo, hi = ci95(diff)
        macro(f"{T}XgbBayesAccDiff", f"{diff.mean():.3f}")
        macro(f"{T}XgbAccGapRef", f"{-diff.mean():.3f}")
        macro(f"{T}XgbAccGapRefCI", f"{-hi:.3f} to {-lo:.3f}")
        macro(f"{T}XgbBayesAccCI", f"{lo:.3f} to {hi:.3f}")
        pf = d[d.target == t].pivot(index="seed", columns="model", values="macro_f1")
        gb = pf[["XGBoost", "LightGBM", "CatBoost"]]
        macro(f"{T}GbmSpread", f"{(gb.mean().max() - gb.mean().min()):.3f}")
        macro(f"{T}GbmSdMax", f"{gb.std().max():.3f}")
        macro(f"{T}Best", names[best[t]])
        dd = pf["OrdinalProbit"] - pf["XGBoost"]
        lo, hi = ci95(dd)
        macro(f"{T}ProbitXgbF", f"{dd.mean():.3f}")
        macro(f"{T}ProbitXgbFCI", f"{lo:.3f} to {hi:.3f}")


# ----------------------------------------------------------------------------- E2 signal
def signal():
    d = pd.read_csv("results/signal.csv")
    bay = d[d.model == "Bayes"].set_index(["target", "r2", "seed"])
    d["exll"] = d.apply(lambda r: r.log_loss - bay.loc[(r.target, r.r2, r.seed), "log_loss"], axis=1)
    m = d.groupby(["target", "r2", "model"])[["accuracy", "macro_f1", "exll"]].mean()
    for t in ["phq", "gad"]:
        T = t.upper()
        for r2, tag in [(0.2, "Low"), (0.4, "Mid"), (0.8, "Ref"), (0.95, "High")]:
            macro(f"{T}BayesAcc{tag}", f"{m.loc[(t, r2, 'Bayes'), 'accuracy']:.2f}")
            macro(f"{T}BayesF{tag}", f"{m.loc[(t, r2, 'Bayes'), 'macro_f1']:.2f}")
        x = m.xs("XGBoost", level="model").loc[t].exll
        p = m.xs("OrdinalProbit", level="model").loc[t].exll
        macro(f"{T}XgbExLLmin", f"{x.min():.3f}"); macro(f"{T}XgbExLLmax", f"{x.max():.3f}")
        macro(f"{T}ProbitExLLmax", f"{p.max():.3f}")
        acc_gap = (m.xs("Bayes", level="model").loc[t].accuracy - m.xs("XGBoost", level="model").loc[t].accuracy)
        macro(f"{T}XgbAccGapMax", f"{acc_gap.max():.3f}")
        macro(f"{T}XgbAccGapMin", f"{acc_gap.min():.3f}")


# ----------------------------------------------------------------------------- E3 attribution figures
def shares_frame(df, method, target):
    q = df[(df.method == method) & (df.target == target)]
    return q[[f"share_{f}" for f in FEATURES]].rename(columns=lambda c: c[6:])


def fig_attr():
    x = pd.read_csv("results/xai.csv")
    k = pd.read_csv("results/kernel.csv") if os.path.exists("results/kernel.csv") else None
    ref = x[(x.lam == 1.0) & (x.r2 == 0.8)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.9), sharey=True)
    order = ["pa_minutes_week", "sleep_hours", "comorbidities", "monthly_income", "living_situation", "gender_female",
             "age", "education_years", "marital_status", "pa_days_week", "pa_minutes_session", "pa_type", "pa_intensity"]
    ypos = np.arange(len(order))
    for ax, t, title in zip(axes, ["phq", "gad"], ["PHQ-9", "GAD-7"]):
        tr = TRUTH[t]
        tot = sum(tr.values())
        n_causal = sum(tr[f] > 0 for f in order)
        ax.axhspan(n_causal - 0.5, len(order) - 0.5, color="#f4f3f0", zorder=0, lw=0)
        ax.barh(ypos, [tr[f] / tot for f in order], height=0.62, color="white", edgecolor=INK, lw=0.9,
                label="True share", zorder=1)
        meths = [("TreeSHAP-interventional", BLUE, "o"), ("TreeSHAP-path_dependent", AQUA, "^"),
                 ("Permutation", ORANGE, "s")]
        if k is not None:
            meths.append(("KernelSHAP", VIOLET, "D"))
        offs = np.linspace(-0.24, 0.24, len(meths))
        for (mname, col, mk), off in zip(meths, offs):
            src = k if mname == "KernelSHAP" else ref
            s = shares_frame(src, mname, t)
            ax.errorbar(s[order].mean().values, ypos + off, xerr=s[order].std().values, fmt=mk, color=col,
                        ms=4, elinewidth=0.8, capsize=0, markeredgecolor="white", markeredgewidth=0.5,
                        label=METHOD_LABEL[mname], zorder=3)
        ax.set_yticks(ypos, labels=[FEATURE_LABELS[f] for f in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Share of total attribution")
        ax.grid(axis="y", visible=False)
        ax.text(0.97, (n_causal + len(order) - 1) / 2, "no direct effect", ha="right", va="center",
                color=INK2, fontsize=7, transform=ax.get_yaxis_transform())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7, handlelength=1.4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("paper/figures/fig3_attributions.pdf")
    # numbers at reference
    for t in ["phq", "gad"]:
        T = t.upper()
        for mname, tag in [("TreeSHAP-interventional", "TSI"), ("TreeSHAP-path_dependent", "TSP"), ("Permutation", "Perm")]:
            q = ref[(ref.method == mname) & (ref.target == t)]
            macro(f"{T}{tag}Proxy", f"{100*q.proxy_share.mean():.0f}")
            macro(f"{T}{tag}ProxySD", f"{100*q.proxy_share.std():.0f}")
            macro(f"{T}{tag}Tau", f"{q.kendall.mean():.2f}")
            macro(f"{T}{tag}Top", f"{q.top3_hit.mean():.2f}")
            macro(f"{T}{tag}Age", f"{100*q.share_age.mean():.1f}")
            macro(f"{T}{tag}PAdays", f"{100*q.share_pa_days_week.mean():.1f}")
            pa_group = q[[f"share_{f}" for f in ["pa_days_week", "pa_minutes_session", "pa_type", "pa_intensity"]]].sum(1)
            macro(f"{T}{tag}PAproxy", f"{100*pa_group.mean():.0f}")
            macro(f"{T}{tag}PAweek", f"{100*q.share_pa_minutes_week.mean():.0f}")
        if k is not None:
            q = k[k.target == t]
            macro(f"{T}KSProxy", f"{100*q.proxy_share.mean():.0f}")
            macro(f"{T}KSTau", f"{q.kendall.mean():.2f}")
            macro(f"{T}KSAge", f"{100*q.share_age.mean():.1f}")
            macro(f"{T}KSsecs", f"{q.secs.mean():.0f}")
        tr = TRUTH[t]; tot = sum(tr.values())
        weak = ["monthly_income", "living_situation", "gender_female"]
        macro(f"{T}TrueTopTwo", f"{100*(tr['pa_minutes_week']+tr['sleep_hours'])/tot:.0f}")
        macro(f"{T}TrueWeak", f"{100*sum(tr[f] for f in weak)/tot:.0f}")
        for mname, tag in [("TreeSHAP-interventional", "TSI"), ("Permutation", "Perm")]:
            q = ref[(ref.method == mname) & (ref.target == t)]
            macro(f"{T}{tag}TopTwo", f"{100*(q.share_pa_minutes_week+q.share_sleep_hours).mean():.0f}")
            macro(f"{T}{tag}Weak", f"{100*q[[f'share_{f}' for f in weak]].sum(1).mean():.0f}")
        macro(f"{T}TruePAweek", f"{100*tr['pa_minutes_week']/tot:.0f}")
        macro(f"{T}TrueSleep", f"{100*tr['sleep_hours']/tot:.0f}")


def fig_recovery():
    x = pd.read_csv("results/xai.csv")
    x = x[x.target == "phq"]
    meths = ["TreeSHAP-interventional", "TreeSHAP-path_dependent", "Permutation"]
    lams, r2s = [0.0, 0.5, 1.0], [0.4, 0.8, 0.95]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.1))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"])
    for j, mname in enumerate(meths):
        for i, (metric, lab, vmin, vmax) in enumerate([("kendall", "Kendall $\\tau_b$ with truth", 0, 1),
                                                         ("proxy_share", "Proxy share", 0, 0.5)]):
            ax = axes[i, j]
            M = np.zeros((3, 3)); S = np.zeros((3, 3))
            for a, lam in enumerate(lams):
                for b_, r2 in enumerate(r2s):
                    q = x[(x.method == mname) & (x.lam == lam) & (x.r2 == r2)][metric]
                    M[a, b_], S[a, b_] = q.mean(), q.std()
            ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for a in range(3):
                for b_ in range(3):
                    val = M[a, b_]
                    txt = f"{val:.2f}" if metric == "kendall" else f"{100*val:.0f}%"
                    dark = (val - vmin) / (vmax - vmin) > 0.55
                    ax.text(b_, a, txt, ha="center", va="center", fontsize=7.5,
                            color="white" if dark else INK)
            ax.set_xticks(range(3)); ax.set_xticklabels([f"{r:g}" for r in r2s])
            ax.set_yticks(range(3)); ax.set_yticklabels([f"{l:g}" for l in lams])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(METHOD_LABEL[mname], fontsize=8, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{lab}\n\ndependence $\\lambda$")
            if i == 1:
                ax.set_xlabel("signal $R^2$")
    fig.tight_layout()
    fig.savefig("paper/figures/fig4_recovery.pdf")
    g = x.groupby(["method", "lam", "r2"])[["kendall", "proxy_share", "top3_hit"]].mean()
    for mname, tag in [("TreeSHAP-interventional", "TSI"), ("TreeSHAP-path_dependent", "TSP"), ("Permutation", "Perm")]:
        macro(f"{tag}ProxyIndepHigh", f"{100*g.loc[(mname, 0.0, 0.95), 'proxy_share']:.0f}")
        macro(f"{tag}ProxyDepLow", f"{100*g.loc[(mname, 1.0, 0.4), 'proxy_share']:.0f}")
        macro(f"{tag}ProxyDepHigh", f"{100*g.loc[(mname, 1.0, 0.95), 'proxy_share']:.0f}")
        macro(f"{tag}ProxyIndepLow", f"{100*g.loc[(mname, 0.0, 0.4), 'proxy_share']:.0f}")
        macro(f"{tag}TauIndepHigh", f"{g.loc[(mname, 0.0, 0.95), 'kendall']:.2f}")
        macro(f"{tag}TauDepLow", f"{g.loc[(mname, 1.0, 0.4), 'kendall']:.2f}")
        macro(f"{tag}TauDepHigh", f"{g.loc[(mname, 1.0, 0.95), 'kendall']:.2f}")
        macro(f"{tag}TopMin", f"{x[x.method == mname].groupby(['lam', 'r2']).top3_hit.mean().min():.2f}")


def fig_curves():
    store = json.load(open("results/curves_reference.json"))
    feats = [("pa_minutes_week", "PA (minutes/week)"), ("sleep_hours", "Sleep (hours/night)"), ("age", "Age (years)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))
    for ax, (f, lab) in zip(axes, feats):
        keys = [k for k in store if k.startswith("phq|") and k.endswith(f"|{f}")]
        for n, kk in enumerate(sorted(keys)):
            r = store[kk]
            ax.plot(r["grid"], r["pd_model"], color=BLUE, lw=1.0, alpha=0.75,
                    label="PDP (XGBoost)" if n == 0 else None)
            ax.plot(r["edges"], r["ale_model"], color=ORANGE, lw=1.0, alpha=0.75, ls="-",
                    label="ALE (XGBoost)" if n == 0 else None)
        r = store[sorted(keys)[0]]
        ax.plot(r["grid"], r["pd_true"], color=INK, lw=2.0, ls="--", label="True effect", zorder=5)
        ax.set_xlabel(lab)
        ax.set_title({"pa_minutes_week": "Physical activity", "sleep_hours": "Sleep",
                      "age": "Age (no direct effect)"}[f], loc="left", fontweight="bold")
    axes[0].set_ylabel("Centred expected\nseverity class (PHQ-9)")
    axes[0].legend(loc="upper right", fontsize=6.8)
    ylo = min(a.get_ylim()[0] for a in axes); yhi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(ylo, yhi)
    fig.tight_layout()
    fig.savefig("paper/figures/fig5_curves.pdf")


def tab_curves():
    c = pd.read_csv("results/curves.csv")
    a = pd.read_csv("results/ablation.csv"); ac = pd.read_csv("results/ablation_curves.csv")
    ref = c[(c.lam == 1.0) & (c.r2 == 0.8) & (c.target == "phq")]
    lines = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             "Predictor & True range & PDP range & ALE range & \\makecell{RMSE vs truth\\\\PDP \\quad ALE} \\\\", "\\midrule"]
    for f, lab in [("pa_minutes_week", "PA, minutes/week"), ("sleep_hours", "Sleep"), ("age", "Age (no direct effect)")]:
        q = ref[ref.feature == f]
        lines.append(f"{lab} & {q.range_true.mean():.2f} & {fmt(q.pd_range_model.mean(), q.pd_range_model.std(), 2)} & "
                     f"{fmt(q.ale_range_model.mean(), q.ale_range_model.std(), 2)} & "
                     f"{q.pd_rmse.mean():.3f} \\quad {q.ale_rmse_vs_truePD.mean():.3f} \\\\")
        T = {"pa_minutes_week": "PA", "sleep_hours": "Sleep", "age": "Age"}[f]
        macro(f"Curve{T}True", f"{q.range_true.mean():.2f}")
        macro(f"Curve{T}PDP", f"{q.pd_range_model.mean():.2f}")
        macro(f"Curve{T}ALE", f"{q.ale_range_model.mean():.2f}")
        macro(f"Curve{T}PDPrmse", f"{q.pd_rmse.mean():.3f}")
        macro(f"Curve{T}ALErmse", f"{q.ale_rmse_vs_truePD.mean():.3f}")
        if q.range_true.mean() > 0:
            macro(f"Curve{T}PDPpct", f"{100*q.pd_range_model.mean()/q.range_true.mean():.0f}")
            macro(f"Curve{T}ALEpct", f"{100*q.ale_range_model.mean()/q.range_true.mean():.0f}")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open("paper/tables/tab_curves.tex", "w").write("\n".join(lines) + "\n")
    # attenuation of the PA curve across the design
    allc = c[(c.feature == "pa_minutes_week") & (c.target == "phq")].copy()
    allc["ratio"] = allc.pd_range_model / allc.range_true
    allc["ratio_ale"] = allc.ale_range_model / allc.range_true
    r = allc.groupby(["lam", "r2"])[["ratio", "ratio_ale"]].mean()
    macro("PAratioIndep", f"{100*r.loc[0.0].ratio.mean():.0f}")
    macro("PAratioDep", f"{100*r.loc[1.0].ratio.mean():.0f}")
    macro("PAratioAleIndep", f"{100*r.loc[0.0].ratio_ale.mean():.0f}")
    macro("PAratioAleDep", f"{100*r.loc[1.0].ratio_ale.mean():.0f}")
    ag = c[(c.feature == "age") & (c.target == "phq")].groupby(["lam", "r2"])[["pd_range_model", "ale_range_model"]].mean()
    macro("AgeSpurIndep", f"{ag.loc[0.0].pd_range_model.mean():.2f}")
    macro("AgeSpurDep", f"{ag.loc[1.0].pd_range_model.mean():.2f}")
    macro("AgeSpurAleIndep", f"{ag.loc[0.0].ale_range_model.mean():.2f}")
    macro("AgeSpurAleDep", f"{ag.loc[1.0].ale_range_model.mean():.2f}")
    # ablation
    lines = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", " & All 13 predictors & Without derived PA variables \\\\", "\\midrule"]
    for mname, lab in [("TreeSHAP-interventional", "Proxy share, TreeSHAP (int.)"), ("Permutation", "Proxy share, permutation")]:
        cells = []
        for fs in ["full", "non-redundant"]:
            q = a[(a.method == mname) & (a.features == fs)]
            cells.append(f"{100*q.proxy_share.mean():.0f}\\% ({100*q.proxy_share.std():.0f})")
            macro(f"Abl{'Full' if fs == 'full' else 'Red'}{'TSI' if 'Tree' in mname else 'Perm'}Proxy", f"{100*q.proxy_share.mean():.0f}")
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    for mname, lab in [("TreeSHAP-interventional", "Kendall $\\tau_b$, TreeSHAP (int.)")]:
        cells = []
        for fs in ["full", "non-redundant"]:
            q = a[(a.method == mname) & (a.features == fs)]
            cells.append(fmt(q.kendall.mean(), q.kendall.std(), 2))
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    for f, lab in [("pa_minutes_week", "PA curve: PDP range / true range"), ("age", "Age curve: PDP range (truth 0)")]:
        cells = []
        for fs in ["full", "non-redundant"]:
            q = ac[(ac.feature == f) & (ac.features == fs)]
            if f == "age":
                cells.append(fmt(q.pd_range_model.mean(), q.pd_range_model.std(), 2))
            else:
                v = q.pd_range_model / q.range_true
                cells.append(f"{100*v.mean():.0f}\\% ({100*v.std():.0f})")
                macro(f"Abl{'Full' if fs == 'full' else 'Red'}PAratio", f"{100*v.mean():.0f}")
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    for fs in ["full", "non-redundant"]:
        q = a[(a.method == "Permutation") & (a.features == fs)]
        macro(f"Abl{'Full' if fs == 'full' else 'Red'}F1", f"{q.perf_macro_f1.mean():.3f}")
    cells = [fmt(a[(a.method == 'Permutation') & (a.features == fs)].perf_macro_f1.mean(),
                 a[(a.method == 'Permutation') & (a.features == fs)].perf_macro_f1.std()) for fs in ["full", "non-redundant"]]
    lines.append("Test macro-F1 & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open("paper/tables/tab_ablation.tex", "w").write("\n".join(lines) + "\n")


def tab_size():
    s = pd.read_csv("results/size.csv")
    ns = sorted(s.n.unique())
    perf = s[s.model.isin(["Bayes", "OrdinalProbit", "XGBoost"])]
    acc = perf.pivot_table(index="n", columns="model", values="accuracy", aggfunc="mean")
    ll = perf.pivot_table(index="n", columns="model", values="log_loss", aggfunc="mean")
    xs = s[s.model == "XAI:TreeSHAP-interventional"].groupby("n")[["kendall", "proxy_share"]].mean()
    xp = s[s.model == "XAI:Permutation"].groupby("n")[["kendall", "proxy_share"]].mean()
    cv = s[s.model == "CURVE:pa_minutes_week"].copy(); cv["ratio"] = cv.pd_range_model / cv.range_true
    cvr = cv.groupby("n").ratio.mean()
    ca = s[s.model == "CURVE:age"].groupby("n").pd_range_model.mean()
    lines = ["\\begin{tabular}{@{}rcccccc@{}}", "\\toprule",
             "$n$ & \\makecell{Accuracy gap\\\\XGBoost vs Bayes} & \\makecell{Excess log-loss\\\\XGBoost} & "
             "\\makecell{Proxy share\\\\TreeSHAP (int.)} & \\makecell{Proxy share\\\\permutation} & "
             "\\makecell{PA curve\\\\PDP / true range} & \\makecell{Age curve\\\\PDP range} \\\\", "\\midrule"]
    for n in ns:
        lines.append(f"{n:,} & {acc.loc[n,'Bayes']-acc.loc[n,'XGBoost']:.3f} & {ll.loc[n,'XGBoost']-ll.loc[n,'Bayes']:.3f} & "
                     f"{100*xs.loc[n,'proxy_share']:.0f}\\% & {100*xp.loc[n,'proxy_share']:.0f}\\% & {100*cvr.loc[n]:.0f}\\% & {ca.loc[n]:.2f} \\\\".replace(",", "\\,"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open("paper/tables/tab_size.tex", "w").write("\n".join(lines) + "\n")
    macro("SizeExLLSmall", f"{ll.loc[ns[0],'XGBoost']-ll.loc[ns[0],'Bayes']:.3f}")
    macro("SizeExLLLarge", f"{ll.loc[ns[-1],'XGBoost']-ll.loc[ns[-1],'Bayes']:.3f}")
    macro("SizeAccGapSmall", f"{acc.loc[ns[0],'Bayes']-acc.loc[ns[0],'XGBoost']:.3f}")
    macro("SizeAccGapLarge", f"{acc.loc[ns[-1],'Bayes']-acc.loc[ns[-1],'XGBoost']:.3f}")
    macro("SizeProxySmall", f"{100*xs.loc[ns[0],'proxy_share']:.0f}")
    macro("SizeProxyLarge", f"{100*xs.loc[ns[-1],'proxy_share']:.0f}")
    macro("SizePermProxySmall", f"{100*xp.loc[ns[0],'proxy_share']:.0f}")
    macro("SizePermProxyLarge", f"{100*xp.loc[ns[-1],'proxy_share']:.0f}")
    macro("SizePAratioSmall", f"{100*cvr.loc[ns[0]]:.0f}")
    macro("SizePAratioLarge", f"{100*cvr.loc[ns[-1]]:.0f}")
    macro("SizeAgeSmall", f"{ca.loc[ns[0]]:.2f}")
    macro("SizeAgeLarge", f"{ca.loc[ns[-1]]:.2f}")


def tab_coupling():
    c = pd.read_csv("results/coupling.csv")
    lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule", "Coupling & Strength & $R^2=0.4$ & $R^2=0.8$ & $R^2=0.95$ \\\\", "\\midrule"]
    for kind, lab in [("independent", "Shared determinants only"), ("correlated noise", "Correlated latent errors"),
                      ("cross-term", "Direct PHQ$\\to$GAD term")]:
        for v in sorted(c[c.coupling == kind].value.unique()):
            q = c[(c.coupling == kind) & (c.value == v)].set_index("r2").spearman
            strength = "--" if kind == "independent" else (f"$\\rho_\\varepsilon={v:g}$" if kind == "correlated noise" else f"$a={v:g}$")
            lines.append(f"{lab if v == sorted(c[c.coupling == kind].value.unique())[0] else ''} & {strength} & "
                         + " & ".join(f"{q.loc[r]:.2f}" for r in [0.4, 0.8, 0.95]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open("paper/tables/tab_coupling.tex", "w").write("\n".join(lines) + "\n")
    q = c[c.coupling == "independent"].set_index("r2").spearman
    macro("CoupIndepLow", f"{q.loc[0.4]:.2f}"); macro("CoupIndepRef", f"{q.loc[0.8]:.2f}"); macro("CoupIndepHigh", f"{q.loc[0.95]:.2f}")


def cohort():
    from scipy.stats import spearmanr
    X = G.sample(n=200_000, seed=123)
    for f, tag in [("pa_minutes_week", "PA"), ("sleep_hours", "Sleep"), ("age", "Age"), ("education_years", "Edu"),
                   ("comorbidities", "Com"), ("pa_days_week", "PAdays")]:
        macro(f"RhoPHQ{tag}", f"{spearmanr(X[f], X['phq9']).statistic:.2f}")
    macro("RhoAgePA", f"{spearmanr(X['age'], X['pa_minutes_week']).statistic:.2f}")
    macro("RhoAgeSleep", f"{spearmanr(X['age'], X['sleep_hours']).statistic:.2f}")
    macro("RhoPASleep", f"{spearmanr(X['pa_minutes_week'], X['sleep_hours']).statistic:.2f}")
    macro("RhoPAdaysPA", f"{spearmanr(X['pa_days_week'], X['pa_minutes_week']).statistic:.2f}")
    macro("RhoPhqGad", f"{spearmanr(X['phq9'], X['gad7']).statistic:.2f}")


def write_numbers():
    with open("paper/numbers.tex", "w") as f:
        for k, v in NUM.items():
            name = "".join(ch for ch in k if ch.isalpha())
            f.write(f"\\newcommand{{\\n{name}}}{{{v}}}\n")
    print(len(NUM), "macros")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["cohort", "bench", "signal", "fig_attr", "fig_recovery", "fig_curves", "tab_curves", "tab_size", "tab_coupling"]
    for s in steps:
        try:
            globals()[s]()
            print("ok", s)
        except FileNotFoundError as e:
            print("skip", s, e)
    write_numbers()
