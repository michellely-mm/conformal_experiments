# Experiments: Conformal Prediction for Continuous Treatments under Hidden Confounding

This code runs the paper's interval procedure (Section 6.1) as written, from the
kernel in eq. (2) through the exact-dose interval in eq. (27). It evaluates the
procedure on two synthetic benchmarks taken from the paper we compare against,
with hidden confounding added.

```
conformal_experiments/
├── run_experiments.py      # runs all experiments, then calls make_figures
├── make_figures.py         # figures (PDF+PNG) + LaTeX tables from results/summary.csv
├── verify.py               # correctness checks for every component (~10 s)
├── cpconf/
│   ├── dgp.py              # data-generating processes (datasets 1 & 2 + hidden U)
│   ├── method.py           # the paper's procedure: eqs (2),(3),(13),(16),(21),(24)-(27),(29)
│   └── replicate.py        # one Monte Carlo replicate: fit, calibrate, evaluate
├── run_on_mac.command      # double-click on macOS
└── run_on_windows.bat
```

## Run it

```bash
python3 -m pip install -r requirements.txt
python3 verify.py                       # all checks must pass
python3 run_experiments.py              # 500 reps, all CPU cores -> results/
python3 run_experiments.py --quick      # 20-rep smoke test -> results_quick/
python3 make_figures.py results         # redraw figures/tables only
```
On macOS you can double-click `run_on_mac.command` instead (if Gatekeeper blocks it,
run `bash run_on_mac.command` in Terminal). The full run takes ~15 min on 2 cores
and ~4 min on 8 cores. Seeds are fixed (`--seed 20261011`), and each setting has
its own `SeedSequence`, so results don't depend on the number of workers.

## Data-generating processes

The observed-data structure is copied from the two synthetic datasets of
Schröder et al. (NeurIPS 2025), `modules/dataset_configs.py` in
github.com/m-schroder/ContinuousCausalCP. In both, X ~ Uniform{1,2,3,4} and the
dose domain is A = [0, 40]:

| | propensity π(a\|x) | response μ(x,a) |
|---|---|---|
| Dataset 1 | step: 0.3·U[0,5x] + 0.7·U[5x,40] | sin((0.1a − 0.5x)π/6) |
| Dataset 2 | N(5x, 10²), truncated to [0,40] | sin((0.1a − 0.1x)π/2) |

Their datasets have no hidden confounding. We add an **unobserved U ∈ {−1,+1}**:

* **Selection.** V | X=x, U=u has density 1 + η·u·cos(2π(v − F₀(a*|x))) on [0,1],
  and A = F₀⁻¹(V | x). Because E[U] = 0, the **observed propensity is exactly theirs**
  (checked by a KS test in `verify.py`). Subjects treated near a* are mostly U = +1.
* **Outcomes.** Y(a) = μ(x,a) + 0.5·U + ε with ε ~ N(0, 0.1²). U and ε are the **same
  at every dose** for a given subject, as Assumption 2 requires.
* **Target.** a* = 20, α = 0.1, δ_K = 0.02, n_train = n_cal = 2000, 500 test subjects
  per replicate.

**Valid sensitivity parameters** (Assumptions 4 and 5). Given (x,a),
P(U = u | x,a) = (1 + u·t)/2 with |t| ≤ η. The observed and interventional outcome
laws are the same Gaussian kernel mixed over U with weights q and (½,½). So:
* by data processing, E[(b−1)² | x,a] = χ²(T‖R) ≤ χ²(½,½ ‖ q) = t²/(1−t²) ≤ η²/(1−η²),
  giving **ρ = η/√(1−η²)**;
* b is a ratio of mixtures with weight ratios ½/q_u ∈ [1/(1+η), 1/(1−η)], giving **Λ = 1/(1−η)**.

`verify.py` confirms both bounds by numerical quadrature. **Both are sharp** in this
design: at a = a*, t = η exactly, and the two outcome components are ~10 sd apart.
So ρ = η/√(1−η²) is the *smallest* valid radius (0.75 at η = 0.6), and in Figure 6
values below it really do violate Assumption 4.

## What the code computes (paper → code)

| Paper | Code |
|---|---|
| φ(x,a) | per-x degree-5 polynomial in a, least squares on the training split |
| π̂(a\|x) | correctly specified parametric MLE (step: shared p; Gaussian: mean b₀+b₁x, common sd) |
| M (Ass. 3) | exact sup over x and a ∈ [0,40] of max(π̂/π, π/π̂), using the known truth |
| g_σ, Z_σ (eq. 2) | Gaussian truncated to [0,40] and renormalised |
| r (eq. 3) | exact for the truncated kernel: Pr{A_σ ∉ B_r} = δ_K |
| L_Y, L_φ (Ass. 2), β = 1 | exact sup of \|∂_a μ\| and \|∂_a φ\| over B_r (endpoints + critical points) |
| w̄(x) (eq. 24) | exact sup over B_r of g_σ/π̂ (Appendix D-style finite candidate set) |
| t_{D_C}(v;p) (eq. 13) | `WeightedCalibration.cutoff`: +∞ atom, ≥ ties, zero weights. Checked against the literal definition |
| α_ρ (eq. 21), α′ (eq. 29) | `alpha_rho`, `alpha_pointwise` |
| interval (eq. 27) | φ(x,a*) ± (t̄ + e_r), with e_r = (L_Y + L_φ)·r |
| n_eff (eq. 31) | reported as a diagnostic |

Every assertion and certificate is checked in `verify.py`. Each replicate also
asserts that the finite/infinite outcome matches Proposition 9.

## Methods compared

Exact-dose target Y_new(a*), nominal level 0.90:
1. **Split CP**: standard unweighted split conformal at a*.
2. **Kernel-weighted CP**: kernel/propensity weights with query weight w₀(x,a*) at
   level α. It corrects for observed confounding and assumes no hidden confounding.
3. **Ours, ρ = 0**: the full Theorem 7 construction with ρ = 0. It accounts for M and
   the exact-dose bridge but ignores hidden confounding.
4. **Ours, second-moment** (Theorem 7) with the valid ρ.
5. **Ours, pointwise** (Corollary 8) with the valid Λ.

Reference (not a competitor): **Oracle (infeasible)**. For each x, it uses the
0.9-quantile of |Y(a*) − φ(x,a*)| under the *true* potential-outcome law (20,000 fresh
draws). It is the narrowest interval centred at φ with exactly 90% coverage. It
measures the cost of the guarantee: the `×Or.` column in Table 1 and Figure 8 are
"our width / oracle width".

Auxiliary randomized intervention Y(A_σ), A_σ ~ g_σ, nominal 1 − α_sm = 0.92
(checks Theorem 5): weighted CP at level α_sm without robustness, compared with
the robust levels α_ρ and α′.

## Experiments and outputs (written to `results/` when you run)

| File | Content |
|---|---|
| `fig1_illustration` | one draw: the interval each method returns at a*, against the true Y(a*), for each x |
| `fig2_confounding` | coverage / finite rate / width vs η (main result) |
| `fig3_auxiliary_theorem5` | Theorem 5 check: coverage of Y(A_σ) vs η |
| `fig4_bandwidth` | σ sweep, including the width split into 2·t̄ and 2·e_r |
| `fig5_calibration_size` | n_c sweep: finiteness (Prop. 9) and the width floor that remains |
| `fig6_sensitivity_misspecification` | true η = 0.6 (valid ρ = 0.75), assumed ρ swept: stress test |
| `fig8_coverage_width_tradeoff` | coverage vs width for every method at η = 0.2/0.4/0.6, compared with the oracle (cost of conservatism) |
| `fig7_propensity_estimation` | cost of estimating π: certified M, α_ρ and width vs training size, compared with oracle π |
| `table1_confounding.tex` | main table: coverage (Monte Carlo SE), finite-interval rate, width given finite, width / oracle width; red = significantly below 0.9 |
| `table2_auxiliary_theorem5.tex` | Theorem 5 table |
| `table3_supplied_constants.tex` | M, ρ, Λ, r, L_Y, L_φ, e_r, α_ρ, α′, W, n_eff, w̄ at the default setting |
| `summary.csv`, `all_replicates.csv.gz` | every number behind the figures |
| `results_overview.md` | readable pivot tables |
| `experiment_settings.json` | all settings, seed, versions, runtime |

Tables need `\usepackage{booktabs}` and `\usepackage[table]{xcolor}`.

## Scope notes (for writing the paper)

* **Coverage** means marginal coverage averaged over replicates (calibration draws)
  and new subjects, exactly as in eq. (28). The band around each line is ±1.96 Monte
  Carlo SE across replicates.
* **Infinite intervals count as covering.** Width is averaged over finite intervals
  only, and the finite-interval rate is always reported next to it (Section 6).
* **How coverage is computed.** In each replicate, fresh train / calibration / test
  samples are drawn independently. Coverage is the fraction of the 500 test subjects
  whose *true* Y_new(a*) (same U and ε, dose set to a*) lies in the interval. The
  reported number is the mean over replicates, with SE = sd/√reps.
* **M is an oracle-aided certificate.** It is computed from the known true propensity
  over the whole support (all x, all a ∈ [0,40]), not from training residuals. Say this
  in the paper: in practice M must be supplied.
* **M, ρ, Λ, L_Y** are *supplied* in the paper. Here they are certified from the
  known simulation truth so that the theorems' hypotheses hold exactly. Figure 6 is
  a deliberate misspecification stress test, not a guarantee.
* **MIMIC-III / TCGA are not included.** MIMIC-III needs credentialed PhysioNet access
  plus the MIMIC-Extract `all_hourly_data.h5`. Even in Schröder et al. the MIMIC
  outcome is semi-synthetic (a linear function of covariates + treatment), and their
  repository contains no TCGA data. A real-data run cannot certify M, ρ or measure
  true potential-outcome coverage. It could only report interval widths under assumed
  sensitivity values.
