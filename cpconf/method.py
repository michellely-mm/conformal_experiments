"""The interval procedure of the paper (Section 6.1), with equation numbers.

Everything that the paper treats as a *supplied* quantity (M, rho / Lambda,
L_Y, L_phi, r, the envelope w_bar) is computed here as a valid certificate
from the known simulation design, so the theorems' hypotheses hold exactly.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .dgp import (A_LO, A_HI, A_STAR, X_LEVELS, StepPropensity, TruncNormPropensity)

# ==========================================================================
# Kernel g_sigma (eq. 2), radius r and tail budget delta_K (eq. 3)
# ==========================================================================
def kernel_Z(sigma, a_star=A_STAR):
    return norm.cdf((A_HI - a_star) / sigma) - norm.cdf((A_LO - a_star) / sigma)


def kernel_pdf(a, sigma, a_star=A_STAR):
    a = np.asarray(a, float)
    inside = (a >= A_LO) & (a <= A_HI)
    return norm.pdf((a - a_star) / sigma) / sigma / kernel_Z(sigma, a_star) * inside


def kernel_peak(sigma, a_star=A_STAR):
    return 1.0 / (kernel_Z(sigma, a_star) * np.sqrt(2 * np.pi) * sigma)


def kernel_sample(n, sigma, rng, a_star=A_STAR):
    lo = norm.cdf((A_LO - a_star) / sigma); hi = norm.cdf((A_HI - a_star) / sigma)
    return np.clip(a_star + sigma * norm.ppf(rng.uniform(lo, hi, n)), A_LO, A_HI)


def kernel_radius(sigma, delta_k, a_star=A_STAR):
    """Smallest r with Pr{|A_sigma - a*| > r} <= delta_K for the TRUNCATED kernel.
    For a* in the middle of A=[0,40] and r < 20 the tail is 2*Phi(-r/sigma)/Z."""
    Z = kernel_Z(sigma, a_star)
    if delta_k <= 0:
        return max(a_star - A_LO, A_HI - a_star)
    r = sigma * norm.ppf(1 - delta_k * Z / 2)
    return float(min(r, min(a_star - A_LO, A_HI - a_star)))


# ==========================================================================
# Calibration levels: eq. (21) second-moment and eq. (29) pointwise
# ==========================================================================
def alpha_rho(alpha, delta_k, M, rho):
    """Eq. (21): root of p + rho*sqrt(p(1-p)) = u on the increasing branch."""
    u = (alpha - delta_k) / M ** 2
    if rho == 0:
        return u
    return 2 * u * u / (2 * u + rho * rho + rho * np.sqrt(rho * rho + 4 * u * (1 - u)))


def alpha_pointwise(alpha, delta_k, M, lam):
    """Eq. (29)."""
    return (alpha - delta_k) / (M ** 2 * lam)


# ==========================================================================
# Weighted conformal cutoff, eq. (13), vectorised over query weights v
# ==========================================================================
class WeightedCalibration:
    """Pre-sorts calibration scores so t_{D_C}(v; p) is O(log n) per query."""

    def __init__(self, scores, weights):
        order = np.argsort(scores, kind="stable")
        self.s = np.asarray(scores, float)[order]
        self.cw = np.cumsum(np.asarray(weights, float)[order])
        self.W = float(self.cw[-1]) if self.cw.size else 0.0

    def cutoff(self, v, p):
        """t_{D_C}(v;p) for an array of query weights v.  +inf atom, '>=' ties.
        Finite iff W >= (1-p)/p * v (Proposition 9)."""
        v = np.atleast_1d(np.asarray(v, float))
        out = np.full(v.shape, np.inf)
        ok = np.isfinite(v) & (self.W > 0)
        target = (1 - p) * (self.W + v[ok])
        # tiny relative tolerance so the equality case of Prop. 9 is finite
        idx = np.searchsorted(self.cw, target * (1 - 1e-12), side="left")
        res = np.full(idx.shape, np.inf)
        fin = idx < self.s.size
        res[fin] = self.s[idx[fin]]
        out[ok] = res
        return out


def is_finite_prop9(W, v, p):
    return (W > 0) & np.isfinite(v) & (W * (1 + 1e-12) >= (1 - p) / p * v)


# ==========================================================================
# Outcome predictor phi(x,a): per-x degree-5 polynomial in the dose
# ==========================================================================
class PolyPredictor:
    DEG = 5

    def fit(self, x, a, y):
        self.coef = {}
        for lev in X_LEVELS:
            m = x == lev
            s = (a[m] - A_STAR) / 20.0
            V = np.vander(s, self.DEG + 1)
            self.coef[int(lev)] = np.linalg.lstsq(V, y[m], rcond=None)[0]
        return self

    def predict(self, x, a):
        x = np.asarray(x); a = np.broadcast_to(np.asarray(a, float), x.shape)
        out = np.empty(x.shape)
        for lev in X_LEVELS:
            m = x == lev
            if m.any():
                out[m] = np.polyval(self.coef[int(lev)], (a[m] - A_STAR) / 20.0)
        return out

    def lipschitz(self, a_lo, a_hi):
        """Exact L_phi = sup_{x, a in [a_lo,a_hi]} |d phi/da| (beta = 1).
        Max of |p'| on an interval is attained at an endpoint or a root of p''."""
        s_lo, s_hi = (a_lo - A_STAR) / 20.0, (a_hi - A_STAR) / 20.0
        best = 0.0
        for c in self.coef.values():
            d1 = np.polyder(c); d2 = np.polyder(d1)
            cand = [s_lo, s_hi]
            for rt in np.roots(d2) if d2.size > 1 else []:
                if abs(rt.imag) < 1e-12 and s_lo <= rt.real <= s_hi:
                    cand.append(rt.real)
            best = max(best, float(np.max(np.abs(np.polyval(d1, cand)))) / 20.0)
        return best


# ==========================================================================
# Propensity estimation pi_hat, certified M (Assumption 3) and envelope (24)
# ==========================================================================
def fit_propensity(prop_true, x, a):
    """Correctly specified parametric maximum-likelihood fit of pi(a|x) on the
    training split.
      step     : mixing weight p shared across x (1 parameter; MLE = mean 1{A < 5X})
      gaussian : mean b0 + b1*x, common sd, truncated to [0,40] (3 parameters)
    """
    if isinstance(prop_true, StepPropensity):
        p = float(np.clip(np.mean(a < 5.0 * x), 0.02, 0.98))
        return StepPropensity({int(l): p for l in X_LEVELS})
    if isinstance(prop_true, TruncNormPropensity):
        def nll(th):
            m, s = th[0] + th[1] * x, np.exp(th[2])
            Z = norm.cdf((A_HI - m) / s) - norm.cdf((A_LO - m) / s)
            return np.sum(np.log(s) + 0.5 * ((a - m) / s) ** 2 + np.log(np.maximum(Z, 1e-300)))
        b1, b0 = np.polyfit(x, a, 1)
        th0 = np.array([b0, b1, np.log(np.std(a - b0 - b1 * x))])
        r = minimize(nll, th0, method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-10, maxiter=6000))
        b0, b1, s = r.x[0], r.x[1], float(np.exp(r.x[2]))
        return TruncNormPropensity({int(l): (float(b0 + b1 * l), s) for l in X_LEVELS})
    raise TypeError(prop_true)


def certified_M(prop_hat, prop_true):
    """Exact M = sup over x and a in A=[0,40] (the support of g_sigma) of
    max(c, 1/c), c = pi_hat/pi.  Computed from the known true propensity."""
    worst = 0.0
    if isinstance(prop_true, StepPropensity):
        for lev in X_LEVELS:
            ph, pt = prop_hat.p[int(lev)], prop_true.p[int(lev)]
            worst = max(worst, abs(np.log(ph / pt)), abs(np.log((1 - ph) / (1 - pt))))
        return float(np.exp(worst))
    for lev in X_LEVELS:
        cand = [A_LO, A_HI]
        (mh, sh), (mt, st) = prop_hat.par[int(lev)], prop_true.par[int(lev)]
        q = -0.5 / sh ** 2 + 0.5 / st ** 2                  # log c is quadratic in a
        if abs(q) > 1e-14:
            vtx =(mh / sh ** 2 - mt / st ** 2) / (1 / sh ** 2 - 1 / st ** 2)
            if A_LO <= vtx <= A_HI:
                cand.append(vtx)
        cand = np.array(cand)
        xx = np.full(cand.shape, float(lev))
        lc = np.log(prop_hat.pdf(cand, xx)) - np.log(prop_true.pdf(cand, xx))
        worst = max(worst, float(np.max(np.abs(lc))))
    return float(np.exp(worst))


def envelope(prop_hat, sigma, r, a_star=A_STAR):
    """Exact certified envelope w_bar(x) = sup_{a in B_r} g_sigma(a)/pi_hat(a|x)
    (eq. 24; Appendix D style: maximise over finitely many candidate points)."""
    a_lo, a_hi = max(A_LO, a_star - r), min(A_HI, a_star + r)
    out = {}
    for lev in X_LEVELS:
        if isinstance(prop_hat, StepPropensity):
            # pi_hat is constant on [0,b) and [b,40]; on each piece (closure)
            # g_sigma/pi_hat is largest at the point closest to a*.
            b = 5.0 * lev
            p = prop_hat.p[int(lev)]
            vals = []
            for (lo, hi, dens) in [(A_LO, b, p / b), (b, A_HI, (1 - p) / (A_HI - b))]:
                lo2, hi2 = max(lo, a_lo), min(hi, a_hi)
                if lo2 <= hi2:
                    c = min(max(a_star, lo2), hi2)
                    vals.append(float(kernel_pdf(c, sigma, a_star)) / dens)
            out[int(lev)] = float(max(vals))
        else:
            mh, sh = prop_hat.par[int(lev)]
            cand = [a_lo, a_hi]
            q = 1 / sh ** 2 - 1 / sigma ** 2        # log w0 = -(a-a*)^2/2s^2 + (a-m)^2/2sh^2 + c
            if abs(q) > 1e-14:
                vtx = (mh / sh ** 2 - a_star / sigma ** 2) / q
                if a_lo <= vtx <= a_hi:
                    cand.append(vtx)
            cand = np.array(cand)
            w = kernel_pdf(cand, sigma, a_star) / prop_hat.pdf(cand, np.full(cand.shape, float(lev)))
            out[int(lev)] = float(np.max(w))
    return out
