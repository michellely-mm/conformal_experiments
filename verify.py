"""Correctness checks: each implemented object against the paper's definition.
Run:  python verify.py      (about 20 seconds)
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, kstest

from cpconf.dgp import (DATASETS, X_LEVELS, A_LO, A_HI, A_STAR, GAMMA_U, NOISE_SD,
                        sample, valid_rho, valid_lambda)
from cpconf.method import (kernel_pdf, kernel_radius, kernel_sample, alpha_rho, alpha_pointwise,
                           WeightedCalibration, is_finite_prop9, PolyPredictor, fit_propensity,
                           certified_M, envelope)

rng = np.random.default_rng(0)
ok = lambda msg: print(f"  [ok] {msg}")


def brute_cutoff(s, w, v, p):
    """Literal eq. (13): inf{t in R u {+inf}: (sum w 1{S<=t} + v 1{t=inf})/(W+v) >= 1-p}."""
    W = w.sum()
    for t in np.sort(np.unique(s)):
        if w[s <= t].sum() / (W + v) >= 1 - p - 1e-12:
            return t
    return np.inf


print("1. Kernel g_sigma (eq. 2) and radius (eq. 3)")
for sig in (0.25, 1.0, 4.0):
    assert abs(quad(lambda a: kernel_pdf(a, sig), A_LO, A_HI, points=[A_STAR], limit=200)[0] - 1) < 1e-8
    r = kernel_radius(sig, 0.02)
    tail = 1 - quad(lambda a: kernel_pdf(a, sig), A_STAR - r, A_STAR + r, limit=200)[0]
    assert tail <= 0.02 + 1e-9 and tail > 0.019
    s = kernel_sample(200000, sig, rng)
    assert abs(np.mean(np.abs(s - A_STAR) > r) - 0.02) < 0.002
ok("normalised on [0,40]; Pr{A_sigma outside B_r} = delta_K exactly; sampler matches")

print("2. Weighted cutoff (eq. 13), ties, zero weights, Proposition 9, Lemma 6")
for trial in range(3000):
    n = rng.integers(1, 15)
    s = rng.integers(0, 6, n).astype(float)               # many ties
    w = rng.exponential(1, n) * (rng.uniform(size=n) > 0.2)  # some zero weights
    v = rng.exponential(2); p = rng.uniform(0.02, 0.5)
    cal = WeightedCalibration(s, w)
    t = cal.cutoff(v, p)[0]
    tb = brute_cutoff(s, w, v, p)
    assert t == tb, (s, w, v, p, t, tb)
    assert np.isfinite(t) == bool(is_finite_prop9(w.sum(), v, p))
    assert cal.cutoff(v * 1.7, p)[0] >= t                  # monotone in v
ok("matches the literal definition on 3000 random cases incl. ties / zero weights")
s = rng.normal(size=999)
for a in (0.05, 0.1, 0.2):
    k = int(np.ceil((999 + 1) * (1 - a)))
    assert WeightedCalibration(s, np.ones(999)).cutoff(1.0, a)[0] == np.sort(s)[k - 1]
ok("unit weights reproduce the standard split-conformal order statistic")

print("3. Calibration levels, eqs. (21) and (29)")
for rho in (0.0, 0.1, 0.5, 1.0, 3.0):
    for M in (1.0, 1.3):
        p = alpha_rho(0.1, 0.02, M, rho); u = 0.08 / M ** 2
        assert 0 < p <= u + 1e-15 and abs(p + rho * np.sqrt(p * (1 - p)) - u) < 1e-12
        q = np.linspace(0, p, 200)
        assert np.all(q + rho * np.sqrt(q * (1 - q)) <= u + 1e-12)
assert alpha_pointwise(0.1, 0.02, 1.0, 2.0) == 0.04
ok("alpha_rho solves p + rho sqrt(p(1-p)) = u on the increasing branch, <= u")

print("4. Hidden confounding keeps the observed propensity pi(a|x) = pi0(a|x)")
for key, ds in DATASETS.items():
    d = sample(ds, 40000, 0.8, rng)
    for lev in X_LEVELS:
        m = d["X"] == lev
        pv = kstest(ds.prop.cdf(d["A"][m], np.full(m.sum(), lev)), "uniform").pvalue
        assert pv > 1e-3, (key, lev, pv)
        # ...while U is strongly selected near a*
    near = np.abs(d["A"] - A_STAR) < 1.0
    assert d["U"][near].mean() > 0.6
ok("KS test: A|X has the reference paper's propensity; U over-represented near a*")

print("5. Sensitivity bounds hold: E[(b-1)^2|x,a] <= rho^2 and b in [1/Lambda, Lambda]")
for key, ds in DATASETS.items():
    for eta in (0.2, 0.6, 0.8):
        rho, lam = valid_rho(eta), valid_lambda(eta)
        for lev in X_LEVELS:
            for a in (0.0, 7.0, A_STAR, 31.0, 40.0):
                v = ds.prop.cdf(a, lev); vs = ds.prop.cdf(A_STAR, lev)
                t = eta * np.cos(2 * np.pi * (v - vs))
                q = {+1: (1 + t) / 2, -1: (1 - t) / 2}
                mu = ds.outcome.mu(lev, a)
                f = lambda y, wts: sum(wts[u] * norm.pdf(y, mu + GAMMA_U * u, NOISE_SD) for u in (1, -1))
                half = {1: 0.5, -1: 0.5}
                b = lambda y: f(y, half) / f(y, q)
                lo, hi = mu - 2 * GAMMA_U - 8 * NOISE_SD, mu + 2 * GAMMA_U + 8 * NOISE_SD
                chi2 = quad(lambda y: (b(y) - 1) ** 2 * f(y, q), lo, hi, limit=200, points=[mu - .5, mu + .5])[0]
                assert chi2 <= rho ** 2 + 1e-9
                ys = np.linspace(lo, hi, 400)
                assert np.all(b(ys) <= lam + 1e-9) and np.all(b(ys) >= 1 / lam - 1e-9)
ok("exact chi-square divergence and likelihood-ratio range checked by quadrature")
# sharpness: at a = a*, t = eta exactly and the components are ~10 sd apart, so the
# bound is attained -> rho = eta/sqrt(1-eta^2) is the SMALLEST valid radius here.
for eta in (0.4, 0.6):
    q = {1: (1 + eta) / 2, -1: (1 - eta) / 2}
    f = lambda y, wts: sum(wts[u] * norm.pdf(y, GAMMA_U * u, NOISE_SD) for u in (1, -1))
    chi2 = quad(lambda y: (f(y, {1: .5, -1: .5}) / f(y, q) - 1) ** 2 * f(y, q), -2, 2,
                points=[-.5, 0, .5], limit=500)[0]
    assert abs(np.sqrt(chi2) - valid_rho(eta)) < 1e-4
ok("rho = eta/sqrt(1-eta^2) is attained (sharp), so rho < 0.75 at eta = 0.6 violates Assumption 4")

print("6. Certificates: M (Assumption 3), envelope (eq. 24), L_Y and L_phi (Assumption 2)")
grid = np.linspace(A_LO, A_HI, 40001)
for key, ds in DATASETS.items():
    for n in (250, 2000):
        d = sample(ds, n, 0.4, rng)
        ph = fit_propensity(ds.prop, d["X"], d["A"])
        M = certified_M(ph, ds.prop)
        phi = PolyPredictor().fit(d["X"], d["A"], d["Y"])
        for sig in (0.5, 2.0):
            r = kernel_radius(sig, 0.02)
            wb = envelope(ph, sig, r)
            br = grid[np.abs(grid - A_STAR) <= r]
            L_phi = phi.lipschitz(A_STAR - r, A_STAR + r)
            L_Y = ds.outcome.lipschitz(A_STAR - r, A_STAR + r)
            for lev in X_LEVELS:
                xx = np.full(grid.shape, float(lev))
                c = ph.pdf(grid, xx) / ds.prop.pdf(grid, xx)
                assert c.max() <= M * (1 + 1e-9) and (1 / c).max() <= M * (1 + 1e-9)
                w0 = kernel_pdf(br, sig) / ph.pdf(br, np.full(br.shape, float(lev)))
                assert w0.max() <= wb[int(lev)] * (1 + 1e-9)
                xb = np.full(br.shape, float(lev))
                dphi = np.abs(np.diff(phi.predict(xb, br)) / np.diff(br))
                dmu = np.abs(np.diff(ds.outcome.mu(xb, br)) / np.diff(br))
                assert dphi.max() <= L_phi * (1 + 1e-6) and dmu.max() <= L_Y * (1 + 1e-6)
ok("dense-grid checks never exceed the certified constants")

print("7. Consistency, independence, and the exact-dose target")
from cpconf.dgp import potential_outcome
for key, ds in DATASETS.items():
    d1, d2 = sample(ds, 5000, 0.6, rng), sample(ds, 5000, 0.6, rng)
    assert np.array_equal(d1["Y"], potential_outcome(ds, d1["X"], d1["A"], d1["U"], d1["eps"]))
    assert abs(np.corrcoef(d1["eps"], d2["eps"])[0, 1]) < 0.05      # independent draws
    # Y(a*) differs from observed Y whenever A != a*: coverage must use Y(a*)
    ystar = potential_outcome(ds, d1["X"], A_STAR, d1["U"], d1["eps"])
    assert np.mean(np.abs(ystar - d1["Y"]) > 1e-9) > 0.99
ok("Y = Y(A); train/cal/test are separate independent draws; target uses Y(a*), not Y")

print("\nAll checks passed.")
