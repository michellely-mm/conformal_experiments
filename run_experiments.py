"""Run every experiment of the paper's Experiments section.

    python run_experiments.py                 # full run (500 reps, all cores)
    python run_experiments.py --quick         # 2-minute smoke test (20 reps)
    python run_experiments.py --reps 500 --out results_500

Outputs go to <out>/ : raw CSV, summary CSV, figures (PDF + PNG), LaTeX tables.
"""
import argparse
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cpconf.dgp import DATASETS, valid_rho, valid_lambda, GAMMA_U, NOISE_SD, A_STAR
from cpconf.replicate import run_replicate

DEFAULTS = dict(eta=0.4, sigma=1.0, n_train=2000, n_cal=2000, n_test=500,
                alpha=0.10, delta_k=0.02)


def experiment_grid():
    """(experiment, setting, overrides) triples.  One parameter varies at a time."""
    g = []
    for eta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        g.append(("confounding", eta, dict(eta=eta)))
    for s in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        g.append(("bandwidth", s, dict(sigma=s)))
    for n in (250, 500, 1000, 2000, 4000, 8000, 16000):
        g.append(("calibration_size", n, dict(n_cal=n)))
    for rho in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):   # true eta = 0.6 -> valid rho = 0.75
        g.append(("misspecification", rho, dict(eta=0.6, rho_assumed=rho)))
    for n in (250, 500, 1000, 2000, 4000, 8000):
        g.append(("propensity_estimated", n, dict(n_train=n)))
        g.append(("propensity_oracle", n, dict(n_train=n, oracle_propensity=True)))
    return g


def _job(args):
    ds_key, exp, setting, overrides, reps, seed_entropy = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    cfg = {**DEFAULTS, **overrides}
    rows = []
    for rep in range(reps):
        for r in run_replicate(DATASETS[ds_key], rng, **cfg):
            r.update(dataset=ds_key, experiment=exp, setting=float(setting), rep=rep,
                     eta=cfg["eta"], sigma=cfg["sigma"], n_cal=cfg["n_cal"], n_train=cfg["n_train"])
            rows.append(r)
    return rows


def summarize(raw):
    keys = ["dataset", "experiment", "setting", "target", "method"]
    g = raw.groupby(keys, sort=False)
    out = g.agg(coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
                n_reps=("coverage", "size"), finite_rate=("finite_rate", "mean"),
                width=("width", "mean"), width_sd=("width", "std"),
                width_median=("width", "median"),
                cutoff_part=("cutoff_part", "mean"), inflation_part=("inflation_part", "mean"),
                nominal=("nominal", "first"),
                M=("M", "mean"), rho=("rho", "first"), Lambda=("Lambda", "first"),
                r=("r", "first"), e_r=("e_r", "mean"), L_Y=("L_Y", "first"), L_phi=("L_phi", "mean"),
                p_sm=("p_sm", "mean"), p_pw=("p_pw", "mean"), W=("W", "mean"),
                n_eff=("n_eff", "mean"), v_bar=("v_bar_mean", "mean"),
                finite_threshold_W=("finite_threshold_W", "mean")).reset_index()
    out["coverage_se"] = out.coverage_sd / np.sqrt(out.n_reps)
    out["width_se"] = out.width_sd / np.sqrt(out.n_reps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--quick", action="store_true", help="20 reps smoke test")
    ap.add_argument("--datasets", nargs="+", default=["d1", "d2"])
    args = ap.parse_args()
    if args.quick:
        args.reps = 20
        if args.out == "results":
            args.out = "results_quick"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    jobs = []
    grid = experiment_grid()
    for di, ds in enumerate(args.datasets):
        for gi, (exp, setting, ov) in enumerate(grid):
            jobs.append((ds, exp, setting, ov, args.reps, [args.seed, di, gi]))

    t0 = time.time()
    rows = []
    print(f"Running {len(jobs)} settings x {args.reps} reps on {args.workers} worker(s) ...", flush=True)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(_job, jobs, chunksize=1)):
                rows += res
                print(f"  [{i + 1:3d}/{len(jobs)}] {jobs[i][0]} {jobs[i][1]:<22s} {jobs[i][2]:g}", flush=True)
    else:
        for i, j in enumerate(jobs):
            rows += _job(j)
            print(f"  [{i + 1:3d}/{len(jobs)}] {j[0]} {j[1]:<22s} {j[2]:g}", flush=True)

    raw = pd.DataFrame(rows)
    summ = summarize(raw)
    raw.to_csv(out / "all_replicates.csv.gz", index=False, compression="gzip")
    summ.to_csv(out / "summary.csv", index=False)

    settings = dict(defaults=DEFAULTS, reps=args.reps, seed=args.seed, a_star=A_STAR,
                    gamma_U=GAMMA_U, noise_sd=NOISE_SD,
                    valid_rho_at_default_eta=valid_rho(DEFAULTS["eta"]),
                    valid_lambda_at_default_eta=valid_lambda(DEFAULTS["eta"]),
                    grid=[(e, s, o) for e, s, o in grid],
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    runtime_seconds=round(time.time() - t0, 1))
    (out / "experiment_settings.json").write_text(json.dumps(settings, indent=2, default=str))

    from make_figures import make_all
    make_all(out)
    print(f"\nDone in {time.time() - t0:.0f}s.  Results in: {out.resolve()}")


if __name__ == "__main__":
    main()
