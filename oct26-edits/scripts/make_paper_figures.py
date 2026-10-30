"""Draw the paper figures from paper/results/ (run scripts/infer_planet9.py first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_planet9 import OUT, selection_cube  # noqa: E402

from planetix.data import load_etnos  # noqa: E402
from planetix.inference.model import LABELS, PARAMS  # noqa: E402

FIG = OUT.parent / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def fig_orientations():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, cut in zip(axes, ("bb16", "ts14")):
        df = load_etnos(cut)
        S = selection_cube(cut).mean(axis=0)
        n = S.shape[0]
        ax.imshow(S.T * n * n, origin="lower", extent=(0, 360, 0, 360), cmap="Greys", vmin=0, vmax=3.0,
                  aspect="auto", alpha=0.85)
        inner = df["a"] < 250
        ax.scatter(df["varpi"][~inner], df["Omega"][~inner], s=22, c=BLUE, edgecolors="white", lw=0.8,
                   label="a > 250 AU", zorder=3)
        if inner.any():
            ax.scatter(df["varpi"][inner], df["Omega"][inner], s=22, c=ORANGE, edgecolors="white", lw=0.8,
                       marker="s", label="150 < a < 250 AU", zorder=3)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_xticks(range(0, 361, 90))
        ax.set_yticks(range(0, 361, 90))
        ax.set_xlabel(r"longitude of perihelion $\varpi$ [deg]")
        ax.set_title(f"{cut}: N = {len(df)}", fontsize=9, color=INK)
        ax.grid(False)
    axes[0].set_ylabel(r"longitude of node $\Omega$ [deg]")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.text(0.5, -0.06, "grey: mean selection function (darker = more likely to be detected if orbits were isotropic)",
             ha="center", color=INK2, fontsize=8)
    fig.savefig(FIG / "fig1_orientations.png")
    plt.close(fig)


PRIOR_NAMES = {
    "wide_mass_a": "m 0.5-30 M_E, a 200-1500 AU", "narrow_bb21": "m 4-10, a 300-600, i < 30 (BB21-like)",
    "wide_kappa": "wider concentration ranges", "tight_kappa": "tighter concentration ranges",
    "fmax_le_0.5": "confined fraction f_max < 0.5", "fmax_ge_0.5": "confined fraction f_max > 0.5",
    "low_incl": "i9 < 25 deg", "wide_incl": "i9 < 90 deg", "width_w0.1": "transition width w = 0.1",
    "width_w0.4": "transition width w = 0.4",
}


def fig_bayes_factors():
    main, nondet, pri, sel, split, ext = (load(n) for n in ("main", "nondet", "priors", "selection", "split",
                                                            "external_selection"))
    rows = [
        ("Sample definition", [
            ("bb16  a>250, q>30  (N=31)  baseline", main["bb16"]),
            ("ts14  a>150, q>30  (N=62)", main["ts14"]),
            ("detached  a>250, q>40  (N=19)", main["detached"]),
        ]),
        ("Direct-search non-detection added", [
            ("bb16", nondet["bb16"]), ("ts14", nondet["ts14"]), ("detached", nondet["detached"]),
        ]),
        ("Selection function (bb16)", [
            ("none (isotropic detection)", sel["no_selection"]),
            ("footprint radius 1 deg", sel["radius_1deg"]),
            ("footprint radius 4 deg", sel["radius_4deg"]),
            ("footprint from a > 50 AU only", sel["bb16_footprint_a_gt_50"]),
            ("planetix.bias footprint", ext["bb16_bias_footprint"]),
            ("ts14, footprint from a > 50 AU", sel["ts14_footprint_a_gt_50"]),
            ("ts14, planetix.bias footprint", ext["ts14_bias_footprint"]),
        ]),
        ("OSSOS-family ETNOs (N=9, a>150)", [
            ("characterised survey simulator", ext["ossos_characterised"]),
            ("empirical footprint", ext["ossos_objects_footprint"]),
        ]),
        ("Priors (bb16)", [(PRIOR_NAMES.get(k, k), v) for k, v in pri.items() if k != "baseline"]),
        ("Discovery epoch (bb16)", [
            (f"discovered before 2016 (N={split['before_2016']['n_obj']})", split["before_2016"]),
        ]),
    ]
    labels, vals, errs, groups = [], [], [], []
    for g, items in rows:
        groups.append((g, len(labels)))
        for lab, r in items:
            labels.append(lab)
            vals.append(r["ln_bf"])
            errs.append(r["ln_bf_err"])
    y = -np.arange(len(labels)) - np.repeat(np.arange(len(rows)), [len(i) for _, i in rows]) * 1.2
    fig, ax = plt.subplots(figsize=(6.6, 0.24 * len(y) + 1.6))
    for x in (0, np.log(3), np.log(10), np.log(30), np.log(100)):
        ax.axvline(x, color=GRID if x else INK2, lw=0.8, zorder=0)
    for x, name in ((np.log(3), "3"), (np.log(10), "10"), (np.log(30), "30"), (np.log(100), "100")):
        ax.text(x, y.max() + 1.3, f"K = {name}", ha="center", fontsize=7, color=INK2)
    colors = [BLUE if v >= 0 else ORANGE for v in vals]
    ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor=INK2, lw=1, zorder=2)
    ax.scatter(vals, y, c=colors, s=26, zorder=3, edgecolors="white", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    for g, k in groups:
        ax.text(-0.02, y[k] + 0.9, g, transform=ax.get_yaxis_transform(), ha="right", fontsize=8, weight="bold",
                color=INK)
    ax.set_xlabel(r"$\ln K$, Bayes factor planet vs no planet  (positive favours a planet)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(y.min() - 0.8, y.max() + 2.0)
    fig.savefig(FIG / "fig2_bayes_factors.png")
    plt.close(fig)


def fig_calibration():
    runs = load("calibration")["runs"]
    cross = [r for r in load("calibration2")["runs"] if r["cut"] == "bb16" and r["cross"]]
    power = load("power")["runs"] if (OUT / "power.json").exists() else []
    main = load("main")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    def pick(rs, kind, n):
        return np.array([r["ln_bf"] for r in rs if r["kind"] == kind and r["n_obj"] == n])

    bins = np.linspace(-3, 21, 73)
    ax = axes[0]
    null31 = pick(runs, "null", 31)
    ax.hist(null31, bins=bins, color=INK2, alpha=0.55, label=f"no planet ({len(null31)})")
    ax.hist([r["ln_bf"] for r in cross], bins=bins, histtype="step", color=INK, lw=1.2,
            label=f"no planet, wrong selection model ({len(cross)})")
    ref31 = pick(runs, "planet", 31)
    ax.hist(ref31, bins=bins, color=BLUE, alpha=0.5, label=f"reference planet ({len(ref31)})")
    ax.axvline(main["bb16"]["ln_bf"], color=ORANGE, lw=2, label="observed bb16")
    ax.set_xlabel(r"$\ln K$")
    ax.set_ylabel("simulated catalogues, N = 31")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    series = [("no planet", runs + power, "null", INK2), ("reference planet", runs, "planet", BLUE)]
    if power:
        series.append(("planet the bb16 data imply", power, "planet", AQUA))
    for label, rs, kind, c in series:
        sizes = sorted({r["n_obj"] for r in rs if r["kind"] == kind})
        stats = np.array([np.percentile(pick(rs, kind, n), [16, 50, 84]) for n in sizes])
        ax.fill_between(sizes, stats[:, 0], stats[:, 2], color=c, alpha=0.18, lw=0)
        ax.plot(sizes, stats[:, 1], "o-", color=c, lw=2, ms=5, label=label)
    ax.axhline(main["bb16"]["ln_bf"], color=ORANGE, lw=1, ls="--")
    for k, name in ((np.log(10), "K = 10"), (np.log(100), "K = 100")):
        ax.axhline(k, color=GRID, lw=1)
        ax.text(250, k + 0.3, name, fontsize=7, color=INK2, ha="right")
    ax.set_xscale("log")
    ticks = [31, 62, 124, 248]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.minorticks_off()
    ax.set_xlabel("number of ETNOs with a > 250 AU")
    ax.set_ylabel(r"$\ln K$  (median, 16-84%)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_calibration.png")
    plt.close(fig)


def fig_posterior(name="posterior_bb16_nondet"):
    s = np.load(OUT / f"{name}.npy")
    keep = ["log10_m9", "log10_a9", "i9", "varpi9", "Omega9"]
    idx = [PARAMS.index(k) for k in keep]
    x = s[:, idx].copy()
    x[:, 0] = 10 ** x[:, 0]
    x[:, 1] = 10 ** x[:, 1]
    labels = [r"$m_9$ [$M_\oplus$]", r"$a_9$ [AU]", r"$i_9$ [deg]", r"$\varpi_9$ [deg]", r"$\Omega_9$ [deg]"]
    log_axes = {0, 1}
    k = len(keep)
    fig, axes = plt.subplots(k, k, figsize=(7.0, 7.0))
    for r in range(k):
        for c in range(k):
            ax = axes[r, c]
            ax.grid(False)
            if c > r:
                ax.axis("off")
                continue
            xc = np.log10(x[:, c]) if c in log_axes else x[:, c]
            if r == c:
                ax.hist(xc, bins=30, color=BLUE, alpha=0.75)
                ax.set_yticks([])
            else:
                yr = np.log10(x[:, r]) if r in log_axes else x[:, r]
                ax.hist2d(xc, yr, bins=30, cmap="Blues")
            if r == k - 1:
                ax.set_xlabel(labels[c])
            else:
                ax.set_xticklabels([])
            if c == 0 and r > 0:
                ax.set_ylabel(labels[r])
            elif c > 0 or r == 0:
                ax.set_yticklabels([]) if r != c else None
            if c in log_axes and r == k - 1:
                ticks = {0: [1, 3, 10], 1: [300, 500, 1000]}[c]
                ax.set_xticks(np.log10(ticks))
                ax.set_xticklabels(ticks)
            if r in log_axes and c == 0 and r != c:
                ticks = {0: [1, 3, 10], 1: [300, 500, 1000]}[r]
                ax.set_yticks(np.log10(ticks))
                ax.set_yticklabels(ticks)
    fig.suptitle("Planet parameters, bb16 sample, if the planet hypothesis is adopted", fontsize=9, y=0.92)
    fig.savefig(FIG / "fig4_posterior.png")
    plt.close(fig)


def fig_influence():
    loo = load("loo")
    base = load("main")["bb16"]["ln_bf"]
    names = sorted(loo, key=lambda k: base - loo[k])
    infl = np.array([base - loo[k] for k in names])
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(range(len(names)), infl, color=[BLUE if v > 0 else ORANGE for v in infl], height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(r"change in $\ln$ BF when the object is included  (positive: object supports a planet)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig5_influence.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_orientations, fig_bayes_factors, fig_calibration, fig_posterior, fig_influence):
        try:
            f()
            print("ok", f.__name__)
        except (FileNotFoundError, KeyError) as e:
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main()
