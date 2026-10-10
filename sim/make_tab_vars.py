"""Table 1: simulated variables, distribution at the reference configuration and direct effects."""
import sys
import numpy as np
sys.path.insert(0, "sim")
from generator import Config, Generator

g = Generator(Config(lam=1.0, r2_phq=0.8, r2_gad=0.8))
X = g.sample(n=200_000, seed=123)
cp, cg = g.coef["phq"], g.coef["gad"]


def ms(v, d=1):
    return f"{np.mean(v):.{d}f} ({np.std(v):.{d}f})"


def pct(v, levels, names):
    return "; ".join(f"{n} {100*np.mean(v == l):.0f}\\%" for l, n in zip(levels, names))


def b(c, k):
    return f"${c[k]:+.2f}$".replace("+", "+") if k in c else "--"


rows = [
    ("Age (years)", "Truncated normal, 60--90", ms(X["age"]), "--", "--"),
    ("Female sex", "Bernoulli", f"{100*X['gender_female'].mean():.0f}\\%", b(cp, "gender_female"), b(cg, "gender_female")),
    ("Education (years)", "Normal, 0--16", ms(X["education_years"]), "--", "--"),
    ("Monthly income (EUR)", "Log-normal, 300--6000", ms(X["monthly_income"], 0), b(cp, "log_income") + "$^{a}$", b(cg, "log_income") + "$^{a}$"),
    ("Marital status", "Categorical given age, sex", pct(X["marital_status"], [2, 4], ["married", "widowed"]), "--", "--"),
    ("Living situation", "Categorical given marital, age, comorb.", pct(X["living_situation"], [1, 4], ["alone", "institution"]), b(cp, "lives_alone") + "$^{b}$", b(cg, "lives_alone") + "$^{b}$"),
    ("Chronic conditions (count)", "Poisson, 0--12", ms(X["comorbidities"]), b(cp, "comorbidities"), b(cg, "comorbidities")),
    ("Sleep (h/night)", "Normal, 3--10", ms(X["sleep_hours"]), b(cp, "sleep_hours"), b(cg, "sleep_hours")),
    ("PA, minutes/week", "days $\\times$ minutes + noise, 0--500", ms(X["pa_minutes_week"], 0), b(cp, "pa_minutes_week"), b(cg, "pa_minutes_week")),
    ("PA, days/week", "from weekly volume", ms(X["pa_days_week"]), "--", "--"),
    ("PA, minutes/session", "from weekly volume", ms(X["pa_minutes_session"]), "--", "--"),
    ("PA type (6 levels)", "Categorical given minutes/week", f"none {100*np.mean(X['pa_type']==0):.0f}\\%; walking {100*np.mean(X['pa_type']==1):.0f}\\%", "--", "--"),
    ("PA intensity (4 levels)", "Categorical given type", f"light {100*np.mean(X['pa_intensity']==1):.0f}\\%; vigorous {100*np.mean(X['pa_intensity']==3):.0f}\\%", "--", "--"),
    ("PHQ-9 total", "Latent model, Eq.~(2)", ms(X["phq9"]), "", ""),
    ("GAD-7 total", "Latent model, Eq.~(2)", ms(X["gad7"]), "", ""),
]
lines = ["\\begin{tabular}{@{}p{3.1cm}p{3.7cm}p{3.6cm}rr@{}}", "\\toprule",
         "Variable & Generation & Mean (SD) or \\% & $\\beta_{\\text{PHQ}}$ & $\\beta_{\\text{GAD}}$ \\\\", "\\midrule"]
for r in rows:
    if r[0] == "PHQ-9 total":
        lines.append("\\midrule")
    lines.append(" & ".join(r) + " \\\\")
pc = np.bincount(X["phq_class"]) / len(X["phq_class"])
gc = np.bincount(X["gad_class"]) / len(X["gad_class"])
lines += ["\\bottomrule", "\\end{tabular}"]
note = (f"$^{{a}}$Effect of log income. $^{{b}}$Effect of living alone. PHQ-9 classes (minimal to severe): "
          + ", ".join(f"{100*p:.0f}\\%" for p in pc) + ". GAD-7 classes: " + ", ".join(f"{100*p:.0f}\\%" for p in gc)
          + ". Statistics from $2\\times10^5$ simulated participants.")
open("paper/tables/tab_vars.tex", "w").write("\n".join(lines) + "\n")
open("paper/tables/tab_vars_note.tex", "w").write(note + "\n")
print("\n".join(lines))
