import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "sim")
from plotstyle import plt, INK, BLUE, ORANGE, INK2

d = pd.read_csv("results/signal.csv")
r2s = sorted(d.r2.unique())
xpos = {r: i for i, r in enumerate(r2s)}
# excess log-loss over the Bayes classifier on the same test participants
bayes = d[d.model == "Bayes"].set_index(["target", "r2", "seed"])["log_loss"]
d["excess_ll"] = d.apply(lambda r: r.log_loss - bayes.loc[(r.target, r.r2, r.seed)], axis=1)

fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.5), sharex=True)
styles = {"Bayes": dict(color=INK, ls="--", marker=None, label="Bayes classifier (ceiling)"),
          "OrdinalProbit": dict(color=BLUE, ls="-", marker="o", label="Ordinal probit"),
          "XGBoost": dict(color=ORANGE, ls="-", marker="s", label="XGBoost")}
for j, (target, title) in enumerate([("phq", "PHQ-9 (5 classes)"), ("gad", "GAD-7 (4 classes)")]):
    for i, (metric, ylab) in enumerate([("accuracy", "Test accuracy"), ("excess_ll", "Excess log-loss\nover Bayes")]):
        ax = axes[i, j]
        s = d[d.target == target].groupby(["model", "r2"])[metric].agg(["mean", "std"]).reset_index()
        for m, st in styles.items():
            if metric == "excess_ll" and m == "Bayes":
                ax.axhline(0, color=INK, ls="--", lw=1.2)
                continue
            q = s[s.model == m]
            ax.errorbar([xpos[r] for r in q.r2], q["mean"], yerr=q["std"], capsize=2, lw=1.4, elinewidth=0.8,
                        color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"],
                        markeredgecolor="white", markeredgewidth=0.6)
        if i == 0:
            ax.set_title(title, loc="left", fontweight="bold")
        if j == 0:
            ax.set_ylabel(ylab)
        ax.set_xticks(range(len(r2s)))
        ax.set_xticklabels([f"{r:g}" for r in r2s])
axes[0, 0].legend(loc="upper left")
fig.supxlabel(r"Share of latent outcome variance explained by the predictors ($R^2$)", fontsize=8, y=0.01)
fig.align_ylabels(axes[:, 0])
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("paper/figures/fig2_signal.pdf")
print(d[d.model != "Bayes"].groupby(["target", "r2", "model"]).excess_ll.mean().unstack().round(3))
