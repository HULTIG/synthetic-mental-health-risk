"""Synthetic cohort generator with an explicit, known data-generating process.

The predictors are drawn from a Gaussian copula whose latent correlation matrix
is interpolated between independence (lam=0) and a reference structure (lam=1)
that reproduces the dependence reported for the original cohort.  PHQ-9 and
GAD-7 are produced by a latent additive model on *observed* predictors, so the
class probabilities of every participant are available in closed form.  This
gives three kinds of ground truth:

* the direct-effect coefficient of every predictor (zero for proxies),
* the Bayes-optimal class probabilities p(y | x), and
* the interventional effect of any causal predictor on p(y).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Reference dependence structure (latent Gaussian scale).
# Order: age, education, income, comorbidities, pa_volume, sleep
LATENT_NAMES = ["age", "edu", "inc", "com", "pa", "slp"]
_SPEARMAN_REF = np.array([
    [1.00, -0.37, -0.35, 0.66, -0.74, -0.80],
    [-0.37, 1.00, 0.73, -0.33, 0.47, 0.47],
    [-0.35, 0.73, 1.00, -0.28, 0.47, 0.43],
    [0.66, -0.33, -0.28, 1.00, -0.64, -0.70],
    [-0.74, 0.47, 0.47, -0.64, 1.00, 0.87],
    [-0.80, 0.47, 0.43, -0.70, 0.87, 1.00],
])
R_REF = 2.0 * np.sin(np.pi * _SPEARMAN_REF / 6.0)  # Spearman -> Pearson (Gaussian)

# Clinical cut-offs (score ranges, inclusive lower bounds of each class).
PHQ_LOWER = [0, 5, 10, 15, 20]          # minimal, mild, moderate, mod. severe, severe
GAD_LOWER = [0, 5, 10, 15]              # minimal, mild, moderate, severe
PHQ_LABELS = ["Minimal", "Mild", "Moderate", "Moderately severe", "Severe"]
GAD_LABELS = ["Minimal", "Mild", "Moderate", "Severe"]

# Features handed to the learners (the original feature set).
FEATURES = [
    "age", "gender_female", "education_years", "monthly_income", "marital_status",
    "living_situation", "comorbidities", "sleep_hours", "pa_days_week",
    "pa_minutes_session", "pa_minutes_week", "pa_type", "pa_intensity",
]
NOMINAL = ["marital_status", "living_situation", "pa_type"]

# Direct effects on the standardised latent outcome (before rescaling to R^2).
# Predictors not listed have no direct effect: they are proxies at best.
CAUSAL_PHQ = {"pa_minutes_week": -0.42, "sleep_hours": -0.36, "comorbidities": 0.22,
              "log_income": -0.16, "lives_alone": 0.12, "gender_female": 0.08}
CAUSAL_GAD = {"pa_minutes_week": -0.36, "sleep_hours": -0.40, "comorbidities": 0.18,
              "log_income": -0.16, "lives_alone": 0.08, "gender_female": 0.16}

# Map from equation terms to the feature that carries them.
TERM_TO_FEATURE = {"pa_minutes_week": "pa_minutes_week", "sleep_hours": "sleep_hours",
                   "comorbidities": "comorbidities", "log_income": "monthly_income",
                   "lives_alone": "living_situation", "gender_female": "gender_female"}


@dataclass
class Config:
    lam: float = 1.0            # dependence strength (0 = independent predictors)
    r2_phq: float = 0.80        # share of latent PHQ variance explained by predictors
    r2_gad: float = 0.80
    noise_corr: float = 0.0     # correlation between PHQ and GAD latent noise
    gad_on_phq: float = 0.0     # optional direct PHQ->GAD cross term (original design)
    n: int = 3915
    seed: int = 0
    ref_n: int = 400_000        # reference sample for constants and thresholds
    _cache: dict = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------
def _latent(cfg: Config, n: int, rng: np.random.Generator) -> np.ndarray:
    R = (1 - cfg.lam) * np.eye(6) + cfg.lam * R_REF
    L = np.linalg.cholesky(R)
    return rng.standard_normal((n, 6)) @ L.T


def _predictors(z: np.ndarray, rng: np.random.Generator) -> dict:
    n = z.shape[0]
    u = stats.norm.cdf(z)
    out = {}
    # Age: truncated normal 60-90
    a, b = (60 - 74.7) / 5.6, (90 - 74.7) / 5.6
    out["age"] = np.round(stats.truncnorm.ppf(u[:, 0], a, b, loc=74.7, scale=5.6)).astype(int)
    out["gender_female"] = (rng.random(n) < 0.53).astype(int)
    out["education_years"] = np.clip(np.round(7.9 + 2.95 * z[:, 1]), 0, 16).astype(int)
    inc = np.exp(np.log(2600) + 0.40 * z[:, 2])
    out["monthly_income"] = np.round(np.clip(inc, 300, 6000), 2)
    out["comorbidities"] = np.clip(stats.poisson.ppf(u[:, 3], 3.1), 0, 12).astype(int)
    out["sleep_hours"] = np.round(np.clip(6.55 + 1.05 * z[:, 5], 3.0, 10.0), 1)
    # Weekly PA volume and its decomposition into days x minutes.
    vol = np.clip(155 + 100 * z[:, 4], 0, 500)
    sess = np.clip(rng.normal(36, 9, n), 10, 62)
    days = np.where(vol < 5, 0, np.clip(np.round(vol / sess), 1, 7)).astype(int)
    mins = np.where(days == 0, 0, np.clip(np.round(vol / np.maximum(days, 1)), 10, 90)).astype(int)
    out["pa_days_week"] = days
    out["pa_minutes_session"] = mins
    out["pa_minutes_week"] = np.clip(np.round(days * mins + rng.normal(0, 8, n)), 0, 500).astype(int)
    out["pa_minutes_week"][days == 0] = 0
    # Activity type (0 none, 1 walking, 2 light, 3 household/gardening, 4 moderate, 5 vigorous)
    v = out["pa_minutes_week"]
    logits = np.stack([np.zeros(n), 1.2 + 0 * v, 0.2 + 0.002 * v, 0.6 + 0 * v,
                       -1.0 + 0.008 * v, -3.0 + 0.010 * v], 1)
    logits[:, 0] = np.where(v == 0, 50, -50)
    out["pa_type"] = _sample_cat(logits, rng)
    # Intensity follows type
    t = out["pa_type"]
    inten = np.select([t == 0, t <= 3, t == 4], [0, 1, 2], 3)
    flip = rng.random(n) < 0.15
    inten = np.where(flip & (t > 0), np.clip(inten + rng.choice([-1, 1], n), 1, 3), inten)
    out["pa_intensity"] = inten.astype(int)
    # Marital status (1 single, 2 married, 3 divorced, 4 widowed) depends on age and gender
    age_c = (out["age"] - 75) / 6
    fem = out["gender_female"]
    ml = np.stack([np.full(n, -1.6), np.full(n, 1.0) - 0.4 * age_c - 0.3 * fem,
                   np.full(n, -1.4) - 0.2 * age_c, -0.3 + 0.9 * age_c + 0.7 * fem], 1)
    out["marital_status"] = _sample_cat(ml, rng) + 1
    # Living situation (1 alone, 2 partner, 3 family, 4 institution)
    m = out["marital_status"]
    com_c = (out["comorbidities"] - 3.1) / 1.8
    married = (m == 2).astype(float)
    ll = np.stack([0.4 - 3.0 * married + 0.2 * age_c, -4.0 + 6.0 * married,
                   -0.2 + 0.2 * age_c, -3.2 + 0.5 * age_c + 0.5 * com_c], 1)
    out["living_situation"] = _sample_cat(ll, rng) + 1
    return out


def _sample_cat(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    c = p.cumsum(1)
    r = rng.random(len(p))[:, None]
    return (r > c).sum(1)


def _terms(X: dict) -> dict:
    return {"pa_minutes_week": X["pa_minutes_week"].astype(float),
            "sleep_hours": X["sleep_hours"].astype(float),
            "comorbidities": X["comorbidities"].astype(float),
            "log_income": np.log(X["monthly_income"]),
            "lives_alone": (X["living_situation"] == 1).astype(float),
            "gender_female": X["gender_female"].astype(float)}


def _score_pmf(mean: float, sd: float, top: int) -> np.ndarray:
    """Right-skewed score distribution (negative binomial, truncated at top)."""
    var = sd ** 2
    p = mean / var
    r = mean * p / (1 - p)
    k = np.arange(top + 1)
    pmf = stats.nbinom.pmf(k, r, p)
    pmf[-1] += 1 - pmf.sum()
    return pmf


PHQ_PMF = _score_pmf(7.84, 6.75, 27)
GAD_PMF = _score_pmf(5.73, 5.01, 21)


# --------------------------------------------------------------------------
class Generator:
    """Holds the constants of one configuration (computed once on a big reference sample)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = np.random.default_rng(10_000 + int(cfg.lam * 1000))
        z = _latent(cfg, cfg.ref_n, rng)
        X = _predictors(z, rng)
        T = _terms(X)
        self.mu = {k: v.mean() for k, v in T.items()}
        self.sd = {k: v.std() for k, v in T.items()}
        self.coef = {}
        self.noise_sd = {}
        self.thr = {}
        eps = rng.multivariate_normal([0, 0], [[1, cfg.noise_corr], [cfg.noise_corr, 1]], cfg.ref_n)
        for j, (name, b, r2, pmf, lower) in enumerate([
                ("phq", CAUSAL_PHQ, cfg.r2_phq, PHQ_PMF, PHQ_LOWER),
                ("gad", CAUSAL_GAD, cfg.r2_gad, GAD_PMF, GAD_LOWER)]):
            eta = sum(b[k] * (T[k] - self.mu[k]) / self.sd[k] for k in b)
            scale = np.sqrt(r2 / eta.var())
            self.coef[name] = {k: b[k] * scale for k in b}   # standardised direct effects
            self.noise_sd[name] = np.sqrt(1 - r2)
            ystar = eta * scale + self.noise_sd[name] * eps[:, j]
            if name == "gad" and cfg.gad_on_phq:
                ystar = self._mix_gad(ystar, self._ystar_phq_ref)
            else:
                self._ystar_phq_ref = ystar
            # Thresholds on the latent scale that map to each integer score.
            cdf = np.cumsum(pmf)[:-1]
            self.thr[name] = np.quantile(ystar, cdf)          # score s <=> thr[s-1] <= y* < thr[s]
            self._ref_std = getattr(self, "_ref_std", {})
            self._ref_std[name] = ystar.std()

    def _mix_gad(self, y_gad, y_phq):
        a = self.cfg.gad_on_phq
        return np.sqrt(1 - a ** 2) * y_gad + a * y_phq

    # ---- core quantities --------------------------------------------------
    def eta(self, X: dict, target: str) -> np.ndarray:
        T = _terms(X)
        return sum(c * (T[k] - self.mu[k]) / self.sd[k] for k, c in self.coef[target].items())

    def class_thresholds(self, target: str) -> np.ndarray:
        lower = PHQ_LOWER if target == "phq" else GAD_LOWER
        t = self.thr[target]
        return np.array([t[s - 1] for s in lower[1:]])  # latent cut between classes

    def class_proba(self, X: dict, target: str) -> np.ndarray:
        """Exact p(class | x) under the generating model (no GAD cross term)."""
        if target == "gad" and self.cfg.gad_on_phq:
            a = self.cfg.gad_on_phq
            mean = np.sqrt(1 - a ** 2) * self.eta(X, "gad") + a * self.eta(X, "phq")
            sd = np.sqrt((1 - a ** 2) * self.noise_sd["gad"] ** 2 + a ** 2 * self.noise_sd["phq"] ** 2
                         + 2 * a * np.sqrt(1 - a ** 2) * self.cfg.noise_corr
                         * self.noise_sd["gad"] * self.noise_sd["phq"])
        else:
            mean, sd = self.eta(X, target), self.noise_sd[target]
        cuts = self.class_thresholds(target)
        F = stats.norm.cdf((cuts[None, :] - mean[:, None]) / sd)
        F = np.hstack([np.zeros((len(mean), 1)), F, np.ones((len(mean), 1))])
        return np.diff(F, axis=1)

    def sample(self, n: int | None = None, seed: int | None = None):
        cfg = self.cfg
        n = n or cfg.n
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        z = _latent(cfg, n, rng)
        X = _predictors(z, rng)
        eps = rng.multivariate_normal([0, 0], [[1, cfg.noise_corr], [cfg.noise_corr, 1]], n)
        y_phq = self.eta(X, "phq") + self.noise_sd["phq"] * eps[:, 0]
        y_gad = self.eta(X, "gad") + self.noise_sd["gad"] * eps[:, 1]
        if cfg.gad_on_phq:
            y_gad = self._mix_gad(y_gad, y_phq)
        X["phq9"] = np.searchsorted(self.thr["phq"], y_phq, side="right")
        X["gad7"] = np.searchsorted(self.thr["gad"], y_gad, side="right")
        X["phq_class"] = np.searchsorted(PHQ_LOWER, X["phq9"], side="right") - 1
        X["gad_class"] = np.searchsorted(GAD_LOWER, X["gad7"], side="right") - 1
        return X

    def true_importance(self, target: str) -> dict:
        """|standardised direct effect| per learner feature (0 for proxies)."""
        imp = {f: 0.0 for f in FEATURES}
        for term, c in self.coef[target].items():
            imp[TERM_TO_FEATURE[term]] += abs(c)
        return imp
