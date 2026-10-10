# Do explanations recover the truth?

Code and data for the paper *"Do explanations recover the truth? A controlled synthetic evaluation of depression and anxiety risk models for older adults"*.

The generator simulates older adults with 13 predictors (six with a direct effect on depression and anxiety, seven without a direct effect) and PHQ-9 / GAD-7 severity classes. Because the outcome model is explicit, three quantities are known exactly:

- the Bayes-optimal class probabilities of every participant,
- the true importance of every predictor (zero for proxies),
- the interventional effect curve of any predictor.

All learners and explanation methods are scored against these quantities.

## Layout

| Path | Content |
|---|---|
| `sim/generator.py` | Data-generating mechanism (Gaussian copula predictors, latent outcome model, closed-form class probabilities) |
| `sim/common.py` | Learners, metrics, PDP/ALE, recovery statistics |
| `sim/tune.py` | One-off hyper-parameter search (reference configuration, seed 0, training split only) |
| `sim/experiments.py` | Experiments E1-E6 (`bench`, `signal`, `xai`, `kernel`, `ablation`, `size`, `coupling`) |
| `sim/make_tab_vars.py`, `sim/make_outputs.py`, `sim/fig_signal.py` | Build every table, figure and in-text number of the paper |
| `sim/style_audit.py` | Writing-style audit used during manuscript preparation |
| `results/` | Raw results (CSV/JSON) and frozen hyper-parameters |
| `data/reference_cohort_seed0.csv` | One cohort at the reference configuration (n = 3915), with the true class probabilities |

## Reproduce

```bash
pip install -r requirements.txt
python sim/tune.py                      # optional: results/hyperparams.json is provided
bash run_all.sh                         # all experiments (about 2 h on 2 CPU cores)
python sim/make_tab_vars.py
python sim/fig_signal.py
python sim/make_outputs.py              # tables, figures and paper/numbers.tex
```

The scripts expect to be run from the repository root and write to `paper/tables`, `paper/figures` and `paper/numbers.tex`.

## Using the generator

```python
import sys; sys.path.insert(0, "sim")
from generator import Config, Generator

g = Generator(Config(lam=1.0, r2_phq=0.8, r2_gad=0.8))   # dependence and signal
X = g.sample(n=3915, seed=1)                              # dict of numpy arrays
p_true = g.class_proba(X, "phq")                          # exact P(class | x)
truth = g.true_importance("phq")                          # |beta| per predictor, 0 for proxies
```

No real participant data were used.
