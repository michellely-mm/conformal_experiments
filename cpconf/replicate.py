"""One Monte Carlo replicate: fit, calibrate, and evaluate every method."""
import numpy as np

from .dgp import (A_STAR, X_LEVELS, GAMMA_U, NOISE_SD, sample, potential_outcome,
                  valid_rho, valid_lambda)
from .method import (PolyPredictor, fit_propensity, certified_M, envelope, kernel_pdf,
                     kernel_sample, kernel_radius, alpha_rho, alpha_pointwise,
                     WeightedCalibration, is_finite_prop9)

# Display names (fixed order = fixed colours in every figure)
SPLIT = "Split CP"
KWCP = "Kernel-weighted CP"
OURS0 = r"Ours, $\rho=0$"
OURS_SM = "Ours, second-moment (Thm. 7)"
OURS_PW = "Ours, pointwise (Cor. 8)"
ORACLE = "Oracle (infeasible)"
EXACT_METHODS = [SPLIT, KWCP, OURS0, OURS_SM, OURS_PW]
REFERENCE = [ORACLE]          # plotted as a width reference, never a competitor

AUX_NAIVE = "Weighted CP, no robustness"
AUX_SM = "Robust, second-moment (Thm. 5)"
AUX_PW = "Robust, pointwise (Cor. 8)"
AUX_METHODS = [AUX_NAIVE, AUX_SM, AUX_PW]


def run_replicate(ds, rng, *, eta, sigma, n_train, n_cal, n_test, alpha=0.10,
                  delta_k=0.02, rho_assumed=None, oracle_propensity=False,
                  return_intervals=False):
    # ---------------- data ------------------------------------------------
    tr = sample(ds, n_train, eta, rng)
    ca = sample(ds, n_cal, eta, rng)
    te = sample(ds, n_test, eta, rng)          # new subjects (X, U, eps)

    # ---------------- fitted quantities (F_fit) ---------------------------
    phi = PolyPredictor().fit(tr["X"], tr["A"], tr["Y"])
    if oracle_propensity:
        pi_hat, M = ds.prop, 1.0
    else:
        pi_hat = fit_propensity(ds.prop, tr["X"], tr["A"])
        M = certified_M(pi_hat, ds.prop)

    # ---------------- supplied constants ----------------------------------
    rho = valid_rho(eta) if rho_assumed is None else float(rho_assumed)
    lam = valid_lambda(eta)
    r = kernel_radius(sigma, delta_k)
    a_lo, a_hi = A_STAR - r, A_STAR + r
    L_Y = ds.outcome.lipschitz(a_lo, a_hi)
    L_phi = phi.lipschitz(a_lo, a_hi)
    e_r = (L_Y + L_phi) * r                                   # eq. (25), beta = 1
    a_sm = alpha - delta_k
    p_sm = alpha_rho(alpha, delta_k, M, rho)                  # eq. (21)
    p_pw = alpha_pointwise(alpha, delta_k, M, lam)            # eq. (29)
    p_0 = alpha_rho(alpha, delta_k, M, 0.0)

    # ---------------- calibration ------------------------------------------
    scores = np.abs(ca["Y"] - phi.predict(ca["X"], ca["A"]))           # eq. (12)
    w = kernel_pdf(ca["A"], sigma) / pi_hat.pdf(ca["A"], ca["X"])        # w_0, eq. (16)
    cal_w = WeightedCalibration(scores, w)
    cal_1 = WeightedCalibration(scores, np.ones_like(scores))
    W = cal_w.W
    n_eff = W ** 2 / np.sum(w ** 2) if W > 0 else np.nan                 # eq. (31)
    wbar = envelope(pi_hat, sigma, r)                                    # eq. (24)

    # ---------------- exact-dose intervals at a* ---------------------------
    xs = te["X"]
    centre = phi.predict(xs, A_STAR)
    y_star = potential_outcome(ds, xs, A_STAR, te["U"], te["eps"])       # true Y_new(a*)
    v_bar = np.array([wbar[int(v)] for v in xs])
    v_star = kernel_pdf(np.full(xs.shape, A_STAR), sigma) / pi_hat.pdf(np.full(xs.shape, A_STAR), xs)

    # Infeasible oracle: for each x, the 0.9-quantile of |Y(a*) - phi(x,a*)| under the
    # TRUE potential-outcome law (fresh draws of U, eps).  It is the narrowest interval
    # centred at phi with exactly 1-alpha conditional coverage; a yardstick for width.
    n_or = 20000
    u_or = rng.choice([-1.0, 1.0], size=n_or); e_or = rng.normal(0, NOISE_SD, n_or)
    h_or = {}
    for lev in X_LEVELS:
        xo = np.full(n_or, float(lev))
        res = np.abs(potential_outcome(ds, xo, A_STAR, u_or, e_or) - phi.predict(xo, A_STAR))
        h_or[int(lev)] = float(np.quantile(res, 1 - alpha))

    half = {
        ORACLE: np.array([h_or[int(v)] for v in xs]),
        SPLIT: cal_1.cutoff(np.ones_like(xs), alpha),
        KWCP: cal_w.cutoff(v_star, alpha),
        OURS0: cal_w.cutoff(v_bar, p_0) + e_r,                           # eq. (27)
        OURS_SM: cal_w.cutoff(v_bar, p_sm) + e_r,
        OURS_PW: cal_w.cutoff(v_bar, p_pw) + e_r,
    }
    # sanity: Proposition 9 must agree with the computed cutoff
    assert np.array_equal(np.isfinite(half[OURS_SM]), is_finite_prop9(W, v_bar, p_sm))

    diag = dict(M=M, rho=rho, Lambda=lam, r=r, e_r=e_r, L_Y=L_Y, L_phi=L_phi,
                p_sm=p_sm, p_pw=p_pw, p_0=p_0, W=W, n_eff=n_eff,
                kappa_hat=W / n_cal, v_bar_mean=float(np.mean(v_bar)),
                finite_threshold_W=float(np.mean((1 - p_sm) / p_sm * v_bar)))
    rows = []
    for m, h in half.items():
        fin = np.isfinite(h)
        cov = np.abs(y_star - centre) <= h
        cut = h - e_r if m in (OURS0, OURS_SM, OURS_PW) else h
        rows.append(dict(target="exact", method=m, coverage=float(cov.mean()),
                         finite_rate=float(fin.mean()),
                         width=float(np.mean(2 * h[fin])) if fin.any() else np.nan,
                         cutoff_part=float(np.mean(2 * cut[fin])) if fin.any() else np.nan,
                         inflation_part=(2 * e_r if m in (OURS0, OURS_SM, OURS_PW) else 0.0),
                         nominal=1 - alpha, **diag))

    # ---------------- auxiliary randomized intervention (Theorem 5) --------
    a_sig = kernel_sample(n_test, sigma, rng)
    y_sig = potential_outcome(ds, xs, a_sig, te["U"], te["eps"])
    res_sig = np.abs(y_sig - phi.predict(xs, a_sig))
    v_sig = kernel_pdf(a_sig, sigma) / pi_hat.pdf(a_sig, xs)
    for m, p in [(AUX_NAIVE, a_sm), (AUX_SM, p_sm), (AUX_PW, p_pw)]:
        h = cal_w.cutoff(v_sig, p)
        fin = np.isfinite(h)
        rows.append(dict(target="auxiliary", method=m, coverage=float(np.mean(res_sig <= h)),
                         finite_rate=float(fin.mean()),
                         width=float(np.mean(2 * h[fin])) if fin.any() else np.nan,
                         cutoff_part=np.nan, inflation_part=0.0,
                         nominal=1 - a_sm, **diag))

    if return_intervals:
        return rows, dict(x=xs, centre=centre, y_star=y_star, half=half, U=te["U"])
    return rows
