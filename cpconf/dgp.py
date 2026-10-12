"""Synthetic data-generating processes with hidden confounding.

The observed-data structure follows the two synthetic datasets of
Schroeder et al. (NeurIPS 2025, "Conformal prediction for causal effects of
continuous treatments"; code: github.com/m-schroder/ContinuousCausalCP):

  X ~ Uniform{1,2,3,4}                         (single discrete covariate)
  Dataset 1: step propensity      0.3*U[0,5x] + 0.7*U[5x,40],
             concave response     sin((0.1a - 0.5x) pi/6)
  Dataset 2: Gaussian propensity  N(5x, 10^2)  (truncated to A = [0,40]),
             oscillating response sin((0.1a - 0.1x) pi/2)

We extend both with a HIDDEN confounder U in {-1,+1} (never seen by the
analyst), which is what our paper is about:

  * U ~ Rademacher, independent of X.
  * Selection: with F0(.|x) the CDF of the base propensity pi0(.|x), draw
        V | X=x, U=u  with density  1 + eta*u*psi_x(V),  psi_x(v)=cos(2pi(v - v*_x)),
    where v*_x = F0(a*|x), and set A = F0^{-1}(V | x).
    Because E[U]=0, the OBSERVED propensity is exactly pi(a|x) = pi0(a|x):
    the reference paper's propensity is preserved, and U only distorts which
    subjects receive which dose. Near a*, psi = 1, so subjects with U = +1 are
    over-represented among those treated close to the target dose.
  * Potential outcomes  Y(a) = mu(x,a) + gamma*U + eps,  eps ~ N(0, 0.1^2).
    eps and U are the SAME for a subject at every dose (Assumption 2 needs a
    subject-level, not only mean-level, regularity), and Y = Y(A) (consistency).

Valid sensitivity parameters (derived in README.md):
  P(U=u | x,a) = (1 + u*t)/2 with |t| = |eta*psi| <= eta, so by data processing
      E[(b-1)^2 | x,a] = chi2(T_{x,a} || R_{x,a}) <= eta^2 / (1-eta^2)
      =>  rho  = eta / sqrt(1-eta^2)             (Assumption 4)
  and b is a ratio of mixtures with mixing ratios (1/2)/q_u in [1/(1+eta), 1/(1-eta)]
      =>  Lambda = 1 / (1-eta)                   (Assumption 5)
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

A_LO, A_HI = 0.0, 40.0          # treatment domain  \mathcal{A}
A_STAR = 20.0                   # target dose a*
X_LEVELS = np.array([1, 2, 3, 4])
GAMMA_U = 0.5                   # effect of hidden U on the outcome
NOISE_SD = 0.1                  # same noise level as the reference paper
PROP_SD = 10.0                  # sd of the Gaussian propensity (dataset 2)
P_LOW = 0.3                     # mixing weight of the step propensity (dataset 1)


# --------------------------------------------------------------------------
# Base (observed) propensities pi0(a|x), CDFs and inverse CDFs
# --------------------------------------------------------------------------
class StepPropensity:
    """0.3*U[0,b] + 0.7*U[b,40] with b = 5x  (dataset 1)."""
    name = "step"

    def __init__(self, p_low=None):
        # p_low: dict x -> mixing weight (lets us represent a fitted pi_hat)
        self.p = {int(x): (P_LOW if p_low is None else p_low[int(x)]) for x in X_LEVELS}

    @staticmethod
    def brk(x):
        return 5.0 * np.asarray(x, float)

    def pvec(self, x):
        return np.vectorize(lambda z: self.p[int(z)])(np.asarray(x))

    def pdf(self, a, x):
        a = np.asarray(a, float); b = self.brk(x); p = self.pvec(x)
        inside = (a >= A_LO) & (a <= A_HI)
        return np.where(a < b, p / b, (1 - p) / (A_HI - b)) * inside

    def cdf(self, a, x):
        a = np.clip(np.asarray(a, float), A_LO, A_HI); b = self.brk(x); p = self.pvec(x)
        return np.where(a < b, p * a / b, p + (1 - p) * (a - b) / (A_HI - b))

    def ppf(self, v, x):
        v = np.asarray(v, float); b = self.brk(x); p = self.pvec(x)
        return np.where(v < p, v / p * b, b + (v - p) / (1 - p) * (A_HI - b))


class TruncNormPropensity:
    """N(m_x, s_x^2) truncated to [0,40]; truth m_x=5x, s_x=10 (dataset 2)."""
    name = "gaussian"

    def __init__(self, params=None):
        self.par = {int(x): ((5.0 * x, PROP_SD) if params is None else params[int(x)])
                    for x in X_LEVELS}

    def _ms(self, x):
        x = np.asarray(x)
        m = np.vectorize(lambda z: self.par[int(z)][0])(x)
        s = np.vectorize(lambda z: self.par[int(z)][1])(x)
        return m, s

    def pdf(self, a, x):
        a = np.asarray(a, float); m, s = self._ms(x)
        Z = norm.cdf((A_HI - m) / s) - norm.cdf((A_LO - m) / s)
        inside = (a >= A_LO) & (a <= A_HI)
        return norm.pdf((a - m) / s) / (s * Z) * inside

    def cdf(self, a, x):
        a = np.clip(np.asarray(a, float), A_LO, A_HI); m, s = self._ms(x)
        lo = norm.cdf((A_LO - m) / s); hi = norm.cdf((A_HI - m) / s)
        return (norm.cdf((a - m) / s) - lo) / (hi - lo)

    def ppf(self, v, x):
        v = np.asarray(v, float); m, s = self._ms(x)
        lo = norm.cdf((A_LO - m) / s); hi = norm.cdf((A_HI - m) / s)
        return np.clip(m + s * norm.ppf(lo + v * (hi - lo)), A_LO, A_HI)


# --------------------------------------------------------------------------
# Outcome surfaces mu(x,a) and their exact local Lipschitz constants
# --------------------------------------------------------------------------
def _max_abs_cos(lo, hi):
    """max_{t in [lo,hi]} |cos t| (exact)."""
    if np.floor(hi / np.pi) >= np.ceil(lo / np.pi):   # contains a multiple of pi
        return 1.0
    return float(max(abs(np.cos(lo)), abs(np.cos(hi))))


@dataclass
class Outcome:
    name: str
    slope: float      # coefficient of a inside the sine (times scale)
    xcoef: float
    scale: float

    def mu(self, x, a):
        return np.sin((self.slope * np.asarray(a) - self.xcoef * np.asarray(x)) * self.scale)

    def lipschitz(self, a_lo, a_hi):
        """Exact sup_{x, a in [a_lo,a_hi]} |d mu / da|   (gives L_Y, beta = 1)."""
        best = 0.0
        for x in X_LEVELS:
            lo = (self.slope * a_lo - self.xcoef * x) * self.scale
            hi = (self.slope * a_hi - self.xcoef * x) * self.scale
            best = max(best, self.slope * self.scale * _max_abs_cos(lo, hi))
        return best


CONCAVE = Outcome("concave", 0.1, 0.5, np.pi / 6)      # dataset 1
OSCILLATING = Outcome("oscillating", 0.1, 0.1, np.pi / 2)  # dataset 2


@dataclass
class Dataset:
    key: str
    label: str
    prop: object
    outcome: Outcome


DATASETS = {
    "d1": Dataset("d1", "Dataset 1 (step propensity, concave response)", StepPropensity(), CONCAVE),
    "d2": Dataset("d2", "Dataset 2 (Gaussian propensity, oscillating response)", TruncNormPropensity(), OSCILLATING),
}


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def _sample_v(u, vstar, eta, rng):
    """Draw V in [0,1] with density 1 + eta*u*cos(2 pi (v - vstar)) by rejection."""
    n = u.size
    v = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        cand = rng.uniform(0, 1, todo.size)
        dens = 1 + eta * u[todo] * np.cos(2 * np.pi * (cand - vstar[todo]))
        ok = rng.uniform(0, 1 + eta, todo.size) <= dens
        v[todo[ok]] = cand[ok]
        todo = todo[~ok]
    return v


def sample(ds: Dataset, n, eta, rng):
    """Observational sample.  Returns dict with X, A, Y and hidden (U, eps)."""
    x = rng.choice(X_LEVELS, size=n).astype(float)
    u = rng.choice([-1.0, 1.0], size=n)
    eps = rng.normal(0, NOISE_SD, n)
    vstar = ds.prop.cdf(np.full(n, A_STAR), x)
    v = _sample_v(u, vstar, eta, rng)
    a = ds.prop.ppf(v, x)
    y = potential_outcome(ds, x, a, u, eps)
    return dict(X=x, A=a, Y=y, U=u, eps=eps)


def potential_outcome(ds: Dataset, x, a, u, eps):
    """Y(a) for subjects (x,u,eps): the same u and eps at every dose."""
    return ds.outcome.mu(x, a) + GAMMA_U * u + eps


def valid_rho(eta):
    return 0.0 if eta == 0 else eta / np.sqrt(1 - eta ** 2)


def valid_lambda(eta):
    return 1.0 / (1.0 - eta)
