"""Publication figures and LaTeX tables from <results>/summary.csv.

    python make_figures.py results          # re-draw without re-running
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cpconf.dgp import DATASETS, X_LEVELS
from cpconf.replicate import (SPLIT, KWCP, OURS0, OURS_SM, OURS_PW, EXACT_METHODS,
                              AUX_NAIVE, AUX_SM, AUX_PW, AUX_METHODS, ORACLE, run_replicate)

# --------------------------------------------------------------------------
# Style: one fixed colour + marker + dash per method, everywhere
# --------------------------------------------------------------------------
INK, INK2, GRID = "#1f1f1d", "#5b5a56", "#e7e6e2"
BLUE, AQUA, VIOLET, ORANGE, GRAY = "#2a78d6", "#14a06f", "#6250d6", "#eb6834", "#8d8c87"
STYLE = {
    SPLIT:     dict(color=GRAY,   marker="o", ls=(0, (4, 2)), label="Split CP"),
    KWCP:      dict(color=ORANGE, marker="s", ls=(0, (4, 2)), label="Kernel-weighted CP"),
    OURS0:     dict(color=VIOLET, marker="^", ls="-.",        label=r"Ours, $\rho = 0$"),
    OURS_SM:   dict(color=BLUE,   marker="D", ls="-",         label=r"Ours, second-moment $\rho$ (Thm. 7)"),
    OURS_PW:   dict(color=AQUA,   marker="v", ls="-",         label=r"Ours, pointwise $\Lambda$ (Cor. 8)"),
    ORACLE:    dict(color=INK,    marker="",  ls=(0, (1, 1.3)), label="Oracle width (infeasible, 90% exact)"),
    AUX_NAIVE: dict(color=ORANGE, marker="s", ls=(0, (4, 2)), label="Weighted CP, no robustness"),
    AUX_SM:    dict(color=BLUE,   marker="D", ls="-",         label=r"Robust, second-moment $\rho$ (Thm. 5)"),
    AUX_PW:    dict(color=AQUA,   marker="v", ls="-",         label=r"Robust, pointwise $\Lambda$ (Cor. 8)"),
}
DS_SHORT = {"d1": "Dataset 1: step propensity, concave response",
            "d2": "Dataset 2: Gaussian propensity, oscillating response"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "lines.linewidth": 1.7, "lines.markersize": 4.6, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans", "figure.dpi": 110,
})
WIDTH_IN = 6.75   # PMLR / ALT text width
# Widths are averaged over finite intervals only; hide points where fewer than
# 10% of intervals are finite (too few to give a stable average).
MIN_FINITE = 0.10


def _line(ax, d, x, y, m, err=None):
    st = STYLE[m]
    d = d.sort_values(x)
    ax.plot(d[x], d[y], color=st["color"], marker=st["marker"], ls=st["ls"],
            markeredgecolor="white", markeredgewidth=0.6, label=st["label"], zorder=3)
    if err is not None and err in d:
        lo, hi = d[y] - 1.96 * d[err], d[y] + 1.96 * d[err]
        ax.fill_between(d[x], lo, hi, color=st["color"], alpha=0.13, lw=0, zorder=2)


def _nominal(ax, level, label, below=False):
    ax.axhline(level, color=INK, lw=0.9, ls=":", zorder=1)
    ax.annotate(label, xy=(1, level), xycoords=("axes fraction", "data"),
                xytext=(-2, -3 if below else 3), textcoords="offset points", ha="right",
                va="top" if below else "bottom", fontsize=7.3, color=INK2,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8), zorder=6)


def _legend(fig, methods, ncol=None, y=1.0):
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ls=STYLE[m]["ls"],
                      markeredgecolor="white", markeredgewidth=0.6, label=STYLE[m]["label"])
               for m in methods]
    fig.legend(handles=handles, loc="outside upper center",
               ncol=ncol or len(methods), frameon=False, handlelength=2.6, columnspacing=1.4)


def _rowlabel(ax, ds):
    pass   # titles are drawn by _grid2 on each subfigure


def _grid2(ncols, height=4.7):
    """Two rows (Dataset 1, Dataset 2), each a titled subfigure."""
    fig = plt.figure(figsize=(WIDTH_IN, height), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.02)
    axs = []
    for sf, ds in zip(sfs, ["d1", "d2"]):
        sf.suptitle(DS_SHORT[ds], x=0.01, ha="left", fontsize=8.8, fontweight="bold", color=INK)
        axs.append(sf.subplots(1, ncols))
    return fig, np.array(axs)


def _save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_confounding(s, out):
    d0 = s[(s.experiment == "confounding") & (s.target == "exact")]
    fig, axs = _grid2(3)
    for i, ds in enumerate(["d1", "d2"]):
        d = d0[d0.dataset == ds]
        for m in EXACT_METHODS:
            dm = d[d.method == m]
            _line(axs[i, 0], dm, "setting", "coverage", m, "coverage_se")
            _line(axs[i, 1], dm, "setting", "finite_rate", m)
            _line(axs[i, 2], dm[dm.finite_rate >= MIN_FINITE], "setting", "width", m)
        _line(axs[i, 2], d[d.method == ORACLE], "setting", "width", ORACLE)
        _nominal(axs[i, 0], 0.9, r"$1-\alpha = 0.9$")
        axs[i, 0].set(ylabel=r"Coverage of $Y_{\rm new}(a^*)$", ylim=(0.45, 1.02))
        axs[i, 1].set(ylabel="Finite-interval rate", ylim=(-0.04, 1.04))
        axs[i, 2].set(ylabel="Mean width (finite)")
        _rowlabel(axs[i, 0], ds)
        for ax in axs[i]:
            ax.set_xlabel(r"Confounding strength $\eta$")
    _legend(fig, EXACT_METHODS + [ORACLE], ncol=3, y=1.0)
    _save(fig, out, "fig2_confounding")


def fig_auxiliary(s, out):
    d0 = s[(s.experiment == "confounding") & (s.target == "auxiliary")]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.6), layout="constrained", sharey=True)
    for i, ds in enumerate(["d1", "d2"]):
        d = d0[d0.dataset == ds]
        for m in AUX_METHODS:
            _line(axs[i], d[d.method == m], "setting", "coverage", m, "coverage_se")
        _nominal(axs[i], d.nominal.iloc[0], r"$1-\alpha_{\rm sm} = 0.92$", below=True)
        axs[i].set(xlabel=r"Confounding strength $\eta$", ylim=(0.6, 1.01))
        axs[i].set_title(DS_SHORT[ds], loc="left", fontweight="bold", fontsize=8.4)
    axs[0].set_ylabel(r"Coverage of $Y(A_\sigma)$, $A_\sigma\sim g_\sigma$")
    _legend(fig, AUX_METHODS, y=1.0)
    _save(fig, out, "fig3_auxiliary_theorem5")


def fig_bandwidth(s, out):
    d0 = s[(s.experiment == "bandwidth") & (s.target == "exact")]
    fig, axs = _grid2(3)
    ours = [OURS0, OURS_SM, OURS_PW]
    for i, ds in enumerate(["d1", "d2"]):
        d = d0[d0.dataset == ds]
        for m in ours:
            dm = d[d.method == m]
            _line(axs[i, 0], dm, "setting", "finite_rate", m)
            _line(axs[i, 1], dm[dm.finite_rate >= MIN_FINITE], "setting", "width", m)
        for m in (SPLIT, KWCP, ORACLE):
            _line(axs[i, 1], d[d.method == m], "setting", "width", m)
        axs[i, 0].set(ylabel="Finite-interval rate", ylim=(-0.04, 1.04))
        axs[i, 1].set(ylabel="Mean width (finite)")
        # width decomposition for Theorem 7
        dm = d[d.method == OURS_SM].sort_values("setting")
        xs = np.arange(len(dm))
        ok = dm.finite_rate.values >= MIN_FINITE
        cut = np.where(ok, dm.cutoff_part.values, 0)
        inf_ = dm.inflation_part.values
        axs[i, 2].bar(xs, cut, color=BLUE, width=0.72, label=r"$2\,\bar t_{D_C}$ (calibration cutoff)")
        axs[i, 2].bar(xs, inf_, bottom=cut, color="#9cc3ef", width=0.72,
                      edgecolor="white", linewidth=1.0, label=r"$2\,e_r$ (exact-dose inflation)")
        for k in np.where(~ok)[0]:
            axs[i, 2].annotate(r"$\infty$", (xs[k], inf_[k]), ha="center", va="bottom", fontsize=9, color=INK2)
        for k in np.where(ok & (dm.finite_rate.values < 0.995))[0]:
            axs[i, 2].annotate(f"{dm.finite_rate.values[k]:.0%}", (xs[k], cut[k] + inf_[k]),
                               ha="center", va="bottom", fontsize=6.4, color=INK2, xytext=(0, 1),
                               textcoords="offset points")
        axs[i, 2].set_xticks(xs, [f"{v:g}" for v in dm.setting])
        axs[i, 2].set(ylabel="Width, Thm. 7 interval")
        axs[i, 2].grid(axis="x", visible=False)
        top = np.nanmax(cut + inf_)
        axs[i, 2].set_ylim(0, top * 1.42)
        axs[i, 2].legend(loc="upper left", frameon=False, fontsize=7.0, handlelength=1.2)
        _rowlabel(axs[i, 0], ds)
        for ax in axs[i]:
            ax.set_xlabel(r"Kernel bandwidth $\sigma$")
    _legend(fig, EXACT_METHODS + [ORACLE], ncol=3, y=1.0)
    _save(fig, out, "fig4_bandwidth")


def fig_calibration_size(s, out):
    d0 = s[(s.experiment == "calibration_size") & (s.target == "exact")]
    fig, axs = _grid2(3)
    for i, ds in enumerate(["d1", "d2"]):
        d = d0[d0.dataset == ds]
        for m in EXACT_METHODS:
            dm = d[d.method == m]
            _line(axs[i, 0], dm, "setting", "coverage", m, "coverage_se")
            _line(axs[i, 1], dm, "setting", "finite_rate", m)
            _line(axs[i, 2], dm[dm.finite_rate >= MIN_FINITE], "setting", "width", m)
        _line(axs[i, 2], d[d.method == ORACLE], "setting", "width", ORACLE)
        _nominal(axs[i, 0], 0.9, r"$1-\alpha$")
        axs[i, 0].set(ylabel="Coverage", ylim=(0.6, 1.02))
        axs[i, 1].set(ylabel="Finite-interval rate", ylim=(-0.04, 1.04))
        axs[i, 2].set(ylabel="Mean width (finite)")
        _rowlabel(axs[i, 0], ds)
        for ax in axs[i]:
            ax.set_xscale("log")
            ax.set_xticks([250, 1000, 4000, 16000], ["250", "1k", "4k", "16k"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(r"Calibration size $n_c$")
    _legend(fig, EXACT_METHODS + [ORACLE], ncol=3, y=1.0)
    _save(fig, out, "fig5_calibration_size")


def fig_misspecification(s, out):
    d0 = s[s.experiment == "misspecification"]
    fig, axs = _grid2(3)
    rho_valid = 0.6 / np.sqrt(1 - 0.36)
    for i, ds in enumerate(["d1", "d2"]):
        d = d0[d0.dataset == ds]
        da = d[(d.target == "auxiliary") & (d.method == AUX_SM)]
        de = d[(d.target == "exact") & (d.method == OURS_SM)]
        _line(axs[i, 0], da, "setting", "coverage", AUX_SM, "coverage_se")
        _nominal(axs[i, 0], 0.92, r"$1-\alpha_{\rm sm}$")
        axs[i, 0].set(ylabel=r"Coverage of $Y(A_\sigma)$ (Thm. 5)", ylim=(0.7, 1.01))
        _line(axs[i, 1], de, "setting", "finite_rate", OURS_SM)
        axs[i, 1].set(ylabel="Finite-interval rate (Thm. 7)", ylim=(-0.04, 1.04))
        _line(axs[i, 2], de[de.finite_rate >= MIN_FINITE], "setting", "width", OURS_SM)
        axs[i, 2].set(ylabel="Mean width (finite), Thm. 7")
        for ax in axs[i]:
            ax.axvspan(-0.05, rho_valid, color="#f6d9cc", alpha=0.45, lw=0, zorder=0)
            ax.axvline(rho_valid, color=INK2, lw=0.9, ls="--", zorder=1)
            ax.set_xlabel(r"Assumed radius $\rho$")
            ax.set_xlim(-0.05, 1.55)
        axs[i, 0].annotate("smallest valid\n" + r"$\rho = 0.75$", xy=(rho_valid, 0.71), xytext=(3, 0),
                           textcoords="offset points", fontsize=7.0, color=INK2)
        axs[i, 0].annotate("Ass. 4 fails\n" + r"($\rho$ too small)", xy=(0.02, 0.71), fontsize=7.0, color="#a2492a")
        _rowlabel(axs[i, 0], ds)
    _save(fig, out, "fig6_sensitivity_misspecification")


def fig_propensity(s, out):
    est = s[(s.experiment == "propensity_estimated") & (s.target == "exact")]
    ora = s[(s.experiment == "propensity_oracle") & (s.target == "exact")]
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.55), layout="constrained")
    dstyle = {"d1": dict(color=BLUE, marker="D"), "d2": dict(color=ORANGE, marker="s")}
    for ds in ["d1", "d2"]:
        e = est[(est.dataset == ds) & (est.method == OURS_SM)].sort_values("setting")
        o = ora[(ora.dataset == ds) & (ora.method == OURS_SM)].sort_values("setting")
        lab = "Dataset 1" if ds == "d1" else "Dataset 2"
        axs[0].plot(e.setting, e.M, **dstyle[ds], markeredgecolor="white", label=lab)
        axs[1].plot(e.setting, e.p_sm, **dstyle[ds], markeredgecolor="white", label=f"{lab}, estimated")
        axs[1].plot(o.setting, o.p_sm, color=dstyle[ds]["color"], ls=":", lw=1.4, label=f"{lab}, oracle ($M=1$)")
        ef = e[e.finite_rate >= MIN_FINITE]
        axs[2].plot(ef.setting, ef.width, **dstyle[ds], markeredgecolor="white", label=f"{lab}, estimated")
        axs[2].plot(o.setting, o.width, color=dstyle[ds]["color"], ls=":", lw=1.4, label=f"{lab}, oracle ($M=1$)")
    axs[0].axhline(1, color=INK, lw=0.9, ls=":")
    axs[0].set(ylabel=r"Oracle-certified bound $M$")
    axs[1].set(ylabel=r"Calibration level $\alpha_\rho$, eq. (21)")
    axs[2].set(ylabel="Width, Thm. 7 interval")
    hs = [Line2D([], [], color=BLUE, marker="D", markeredgecolor="white", label="Dataset 1"),
          Line2D([], [], color=ORANGE, marker="s", markeredgecolor="white", label="Dataset 2"),
          Line2D([], [], color=INK2, label=r"estimated $\hat\pi$, certified $M$"),
          Line2D([], [], color=INK2, ls=":", label=r"oracle $\pi$ ($M=1$)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=4, frameon=False)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks([250, 1000, 4000], ["250", "1k", "4k"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"Training size for $\hat\pi$")
    _save(fig, out, "fig7_propensity_estimation")


def fig_illustration(out, eta=0.5, sigma=1.0, seed=7):
    """Figure 1: one calibration draw; the interval each method returns at a*,
    against the true potential outcomes Y_new(a*) of 600 new subjects."""
    methods = [SPLIT, KWCP, OURS_SM, OURS_PW]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0), layout="constrained")
    for k, ds in enumerate(["d1", "d2"]):
        rng = np.random.default_rng(seed)
        rows, iv = run_replicate(DATASETS[ds], rng, eta=eta, sigma=sigma, n_train=2000,
                                 n_cal=2000, n_test=600, return_intervals=True)
        ax = axs[k]
        jit = np.random.default_rng(0)
        infinite = []
        for j, lev in enumerate(X_LEVELS):
            m = iv["x"] == lev
            xj = j + jit.uniform(-0.38, -0.05, m.sum())
            col = np.where(iv["U"][m] > 0, "#3d3c39", "#b9b8b2")
            ax.scatter(xj, iv["y_star"][m], s=3.5, c=col, lw=0, zorder=2, rasterized=True)
            c = iv["centre"][m][0]
            for q, meth in enumerate(methods):
                h = iv["half"][meth][m][0]
                xpos = j + 0.04 + 0.085 * q
                st = STYLE[meth]
                if np.isfinite(h):
                    ax.plot([xpos, xpos], [c - h, c + h], color=st["color"], lw=2.4,
                            solid_capstyle="round", zorder=3)
                else:
                    infinite.append((xpos, st["color"]))
            ax.plot([j + 0.02, j + 0.04 + 0.085 * (len(methods) - 1) + 0.02], [c, c],
                    color=INK, lw=0.8, zorder=4)
        lo, hi = ax.get_ylim()
        hi = hi + 0.1 * (hi - lo)
        for xpos, colr in infinite:     # uninformative output: the whole real line
            ax.plot([xpos, xpos], [lo, hi], color=colr, lw=1.2, ls=(0, (1, 1.6)), zorder=1)
            ax.annotate(r"$\mathbb{R}$", (xpos, hi), ha="center", va="top", fontsize=8.5, color=colr,
                        xytext=(0, -1), textcoords="offset points",
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=5)
        ax.set_ylim(lo, hi)
        ax.set_xticks(range(len(X_LEVELS)), [f"$x={v}$" for v in X_LEVELS])
        ax.grid(axis="x", visible=False)
        ax.set_title(DS_SHORT[ds], loc="left", fontweight="bold", fontsize=8.4)
        ax.set_ylabel(r"$Y_{\rm new}(a^*)$ and intervals at $a^*=20$" if k == 0 else "")
    handles = [Line2D([], [], color=STYLE[m]["color"], lw=2.4, label=STYLE[m]["label"]) for m in methods]
    handles += [Line2D([], [], color="#3d3c39", marker="o", ls="", ms=3, label=r"true $Y(a^*)$, hidden $U=+1$"),
                Line2D([], [], color="#b9b8b2", marker="o", ls="", ms=3, label=r"true $Y(a^*)$, hidden $U=-1$")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, columnspacing=1.3)
    _save(fig, out, "fig1_illustration")


# --------------------------------------------------------------------------
# LaTeX tables (booktabs)
# --------------------------------------------------------------------------
def _cov(r):
    return f"{r.coverage:.3f} ({r.coverage_se:.3f})"


def _wid(r):
    if r.finite_rate < MIN_FINITE:
        return r"$\infty$"
    w = f"{r.width:.2f}"
    return w if r.finite_rate > 0.995 else w + rf"$^{{\,{100 * r.finite_rate:.0f}\%}}$"


def table_main(s, out, etas=(0.0, 0.2, 0.4, 0.6, 0.8)):
    """Coverage (MC SE) | finite rate | width given finite | width / oracle width."""
    d = s[(s.experiment == "confounding") & (s.target == "exact")]
    short = {ORACLE: r"\textit{Oracle (infeasible)}", SPLIT: "Split CP", KWCP: "Kernel-weighted CP",
             OURS0: r"Ours, $\rho=0$", OURS_SM: r"Ours, $\rho$ (Thm.~7)", OURS_PW: r"Ours, $\Lambda$ (Cor.~8)"}
    rows_m = [ORACLE] + EXACT_METHODS
    L = [r"\begin{tabular}{ll" + "cccc" * 2 + "}", r"\toprule",
         r" & & \multicolumn{4}{c}{Dataset 1} & \multicolumn{4}{c}{Dataset 2} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"$\eta$ & Method & Cov. (SE) & Finite & Width & $\times$Or. & Cov. (SE) & Finite & Width & $\times$Or. \\",
         r"\midrule"]
    for e in etas:
        for k, m in enumerate(rows_m):
            cells = []
            for ds in ["d1", "d2"]:
                dd = d[(d.dataset == ds) & np.isclose(d.setting, e)]
                r = dd[dd.method == m].iloc[0]
                orw = dd[dd.method == ORACLE].iloc[0].width
                cov = f"{r.coverage:.3f} ({r.coverage_se:.3f})"
                if m != ORACLE and r.coverage + 1.96 * r.coverage_se < 0.9:
                    cov = r"\textcolor{red!70!black}{" + cov + "}"
                fin = f"{r.finite_rate:.2f}"
                if r.finite_rate >= MIN_FINITE:
                    wid, rat = f"{r.width:.2f}", ("--" if m == ORACLE else f"{r.width / orw:.2f}")
                else:
                    wid, rat = r"$\infty$", r"$\infty$"
                cells += [cov, fin, wid, rat]
            lead = f"{e:.1f}" if k == 0 else ""
            L.append(f"{lead} & {short[m]} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if e != etas[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    (out / "table1_confounding.tex").write_text("\n".join(L) + "\n")


def fig_tradeoff(s, out, etas=(0.2, 0.4, 0.6)):
    """Coverage vs width: where each method sits relative to the oracle."""
    d0 = s[(s.experiment == "confounding") & (s.target == "exact")]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.9), layout="constrained", sharey=True)
    msize = {0.2: 4.0, 0.4: 6.0, 0.6: 8.5}
    for i, ds in enumerate(["d1", "d2"]):
        ax = axs[i]
        d = d0[d0.dataset == ds]
        for m in [ORACLE] + EXACT_METHODS:
            st = STYLE[m]
            dm = d[(d.method == m) & d.setting.round(3).isin(etas)].sort_values("setting")
            dm = dm[dm.finite_rate >= MIN_FINITE]
            ax.plot(dm.width, dm.coverage, color=st["color"], lw=0.9, ls=st["ls"], zorder=2)
            for _, r in dm.iterrows():
                full = r.finite_rate >= 0.99
                ax.plot(r.width, r.coverage, marker=st["marker"] or "o", ms=msize[round(r.setting, 1)],
                        color=st["color"], mfc=st["color"] if full else "white",
                        mec=st["color"] if not full else "white", mew=1.1, ls="", zorder=3)
        _nominal(ax, 0.9, r"$1-\alpha$")
        ax.set(xlabel="Mean interval width (finite)", ylim=(0.6, 1.02))
        ax.set_title(DS_SHORT[ds], loc="left", fontweight="bold", fontsize=8.4)
    axs[0].set_ylabel(r"Coverage of $Y_{\rm new}(a^*)$")
    hs = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"] or "o", ls=STYLE[m]["ls"],
                 markeredgecolor="white", label=STYLE[m]["label"]) for m in [ORACLE] + EXACT_METHODS]
    hs += [Line2D([], [], color=INK2, marker="o", ls="", ms=msize[e], label=rf"$\eta={e}$") for e in etas]
    hs += [Line2D([], [], color=INK2, marker="o", ls="", mfc="white", label="hollow: some intervals infinite")]
    fig.legend(handles=hs, loc="outside upper center", ncol=4, frameon=False, columnspacing=1.1)
    _save(fig, out, "fig8_coverage_width_tradeoff")


def table_aux(s, out, etas=(0.0, 0.2, 0.4, 0.6, 0.8)):
    d = s[(s.experiment == "confounding") & (s.target == "auxiliary")]
    L = [r"\begin{tabular}{ll" + "c" * len(etas) + "}", r"\toprule",
         r"Dataset & Method & " + " & ".join(rf"$\eta={e:.1f}$" for e in etas) + r" \\", r"\midrule"]
    for ds in ["d1", "d2"]:
        for k, m in enumerate(AUX_METHODS):
            cells = []
            for e in etas:
                r = d[(d.dataset == ds) & (d.method == m) & np.isclose(d.setting, e)].iloc[0]
                c = f"{r.coverage:.3f}"
                if r.coverage + 1.96 * r.coverage_se < r.nominal:
                    c = r"\textcolor{red!70!black}{" + c + "}"
                cells.append(c)
            L.append(f"{'Dataset ' + ds[1] if k == 0 else ''} & {m} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if ds == "d1" else r"\bottomrule")
    L.append(r"\end{tabular}")
    (out / "table2_auxiliary_theorem5.tex").write_text("\n".join(L) + "\n")


def table_constants(s, out):
    d = s[(s.experiment == "confounding") & (s.target == "exact") & (s.method == OURS_SM)
          & np.isclose(s.setting, 0.4)]
    rows = [("$M$ (certified)", "M", "{:.3f}"), (r"$\rho$", "rho", "{:.3f}"), (r"$\Lambda$", "Lambda", "{:.3f}"),
            ("$r$", "r", "{:.2f}"), ("$L_Y$", "L_Y", "{:.3f}"), (r"$L_\varphi$", "L_phi", "{:.3f}"),
            ("$e_r$", "e_r", "{:.3f}"), (r"$\alpha_\rho$, eq.~(21)", "p_sm", "{:.4f}"),
            (r"$\alpha'$, eq.~(29)", "p_pw", "{:.4f}"), ("$W$", "W", "{:.0f}"),
            (r"$n_{\rm eff}$, eq.~(31)", "n_eff", "{:.0f}"), (r"mean $\bar w(x)$", "v_bar", "{:.2f}"),
            (r"finite iff $W\ge$", "finite_threshold_W", "{:.0f}")]
    L = [r"\begin{tabular}{lcc}", r"\toprule", r"Quantity & Dataset 1 & Dataset 2 \\", r"\midrule"]
    for lab, col, fmt in rows:
        L.append(f"{lab} & " + " & ".join(fmt.format(d[d.dataset == ds][col].iloc[0]) for ds in ["d1", "d2"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "table3_supplied_constants.tex").write_text("\n".join(L) + "\n")


def text_summary(s, out):
    lines = ["# Key numbers (exact-dose target, coverage +- 1.96 SE, mean width)\n"]
    for exp in ["confounding", "bandwidth", "calibration_size", "misspecification"]:
        lines.append(f"\n## {exp}\n")
        d = s[(s.experiment == exp)]
        for ds in ["d1", "d2"]:
            lines.append(f"\n### {DS_SHORT[ds]}\n")
            t = d[d.dataset == ds].pivot_table(index=["target", "method"], columns="setting",
                                               values="coverage", sort=False).round(3)
            lines.append("Coverage\n\n" + t.to_markdown() + "\n")
            t = d[(d.dataset == ds) & (d.target == "exact")].pivot_table(
                index="method", columns="setting", values="width", sort=False).round(2)
            lines.append("\nWidth (finite intervals)\n\n" + t.to_markdown() + "\n")
            t = d[(d.dataset == ds) & (d.target == "exact")].pivot_table(
                index="method", columns="setting", values="finite_rate", sort=False).round(3)
            lines.append("\nFinite-interval rate\n\n" + t.to_markdown() + "\n")
    (out / "results_overview.md").write_text("\n".join(lines))


def make_all(out):
    out = Path(out)
    s = pd.read_csv(out / "summary.csv")
    fig_illustration(out)
    fig_confounding(s, out)
    fig_auxiliary(s, out)
    fig_bandwidth(s, out)
    fig_calibration_size(s, out)
    fig_misspecification(s, out)
    fig_propensity(s, out)
    fig_tradeoff(s, out)
    table_main(s, out)
    table_aux(s, out)
    table_constants(s, out)
    try:
        text_summary(s, out)
    except ImportError:          # to_markdown needs 'tabulate'
        pass
    print("Figures and tables written to", out.resolve())


if __name__ == "__main__":
    make_all(sys.argv[1] if len(sys.argv) > 1 else "results")
