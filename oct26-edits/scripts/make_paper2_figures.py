"""Figures 6-11 for the evidence-history / LSST-forecast paper (run forecast_and_history.py
and nbody_population.py first). Each figure is skipped if its results file is missing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_paper_figures import AQUA, BLUE, FIG, INK, INK2, ORANGE, load  # noqa: E402  (also sets rcParams)
from infer_planet9 import CACHE, OUT, selection_cube  # noqa: E402

from planetix.data import load_etnos  # noqa: E402

JEFFREYS = [(np.log(3), "3"), (np.log(10), "10"), (np.log(30), "30"), (np.log(100), "100")]


def _have(name):
    ok = (OUT / f"{name}.json").exists()
    if not ok:
        print(f"  skip: {name}.json missing")
    return ok


def fig_history():
    if not _have("history"):
        return
    h = load("history")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for cut, c, lab in (("bb16", BLUE, "a > 250 au"), ("ts14", ORANGE, "a > 150 au")):
        rows = h[cut]
        y = [r["year"] for r in rows]
        v = [r["ln_bf"] for r in rows]
        e = [r["ln_bf_err"] for r in rows]
        ax.errorbar(y, v, e, color=c, marker="o", ms=3.5, lw=1.4, capsize=0, label=lab)
        for r in rows[-1:]:
            ax.annotate(f"N={r['n']}", (r["year"], r["ln_bf"]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=c, va="center")
    for lv, lab in JEFFREYS:
        ax.axhline(lv, color=INK2, lw=0.5, ls=":")
        ax.text(2003.2, lv, f"K={lab}", fontsize=6.5, color=INK2, va="bottom")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(2016, color=INK2, lw=0.8, ls="--")
    ax.text(2016.2, ax.get_ylim()[0] + 0.3, "Batygin & Brown\n2016", fontsize=6.5, color=INK2)
    ax.set_xlabel("objects discovered up to year")
    ax.set_ylabel(r"$\ln K$ (planet : no planet)")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(FIG / "fig6_history.png")
    plt.close(fig)


def fig_predictive():
    files = [OUT / "predictive_2015.json"] if (OUT / "predictive_2015.json").exists() else []
    if not files:
        print("  skip: predictive missing")
        return
    p = json.loads(files[0].read_text())
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bins = np.linspace(-8, 10, 61)
    ax.hist(p["sims_null"], bins, color=INK2, alpha=0.45, density=True, label="no planet")
    ax.hist(p["sims_planet"], bins, color=BLUE, alpha=0.55, density=True, label="planet implied by pre-2016 objects")
    ax.axvline(p["ln_k_new_observed"], color=ORANGE, lw=2, label=f"observed ({p['n_test']} objects)")
    ax.set_xlabel(r"$\ln K_\mathrm{new}$ of the post-%d objects" % p["split_year"])
    ax.set_ylabel("density")
    ax.legend(fontsize=6.5, loc="upper right")
    fig.savefig(FIG / "fig7_predictive.png")
    plt.close(fig)


def fig_selmarg():
    if not _have("selmarg"):
        return
    s = load("selmarg")
    al = s["alphas"]
    rows = [s["models"][f"alpha{a:.1f}"] for a in al]
    lk = np.array([r["ln_bf"] for r in rows])
    z0 = np.array([r["ln_z0"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(al, lk, "o-", color=BLUE, ms=3.5, label=r"$\ln K(\alpha)$")
    ax.axhline(s["bma_continuous"]["ln_bf"], color=BLUE, ls="--", lw=1, label="marginalised")
    ax.set_xlabel(r"selection model $\alpha$ (0: baseline footprint, 1: $a>50$ au, 4$^\circ$)")
    ax.set_ylabel(r"$\ln K$")
    ax2 = ax.twinx()
    ax2.plot(al, z0 - z0.max(), "s-", color=AQUA, ms=3, label=r"$\ln Z_0(\alpha)$")
    ax2.set_ylabel(r"$\ln Z_0 - \max$", color=AQUA)
    ax2.grid(False)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig8_selmarg.png")
    plt.close(fig)


def fig_nbody():
    if not _have("nbody_population"):
        return
    nb = load("nbody_population")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for (name, r), c in zip(nb.items(), (BLUE, ORANGE, AQUA)):
        prof = r["profile"]
        a = [np.sqrt(p["a_lo"] * p["a_hi"]) for p in prof]
        pl = r["planet"]
        axes[0].plot(a, [p["R_varpi"] for p in prof], "o-", color=c, ms=3.5,
                     label=f"{name}: {pl['m9']:g} $M_\\oplus$, {pl['a9']:g} au, e={pl['e9']:g}")
        far = [(ai, p) for ai, p in zip(a, prof) if p["R_varpi"] > 0.2]
        axes[1].plot([x for x, _ in far], [(p["mean_dvarpi"] + 180) % 360 - 180 for _, p in far], "o-", color=c, ms=3.5)
    post = OUT / "posterior_bb16_nondet.npy"
    if post.exists():
        from scipy.special import i0e, i1e
        th = np.median(np.load(post), axis=0)
        k = 10 ** th[7]
        axes[0].axhline(th[6] * i1e(k) / i0e(k), color=INK, ls="--", lw=1)
        axes[0].text(160, th[6] * i1e(k) / i0e(k) + 0.02, "implied by the 31 ETNOs (a >> a_crit)", fontsize=6.5, color=INK)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("semi-major axis [au]")
        ax.axvline(250, color=INK2, ls=":", lw=0.8)
    axes[0].set_ylabel(r"perihelion clustering $R(\varpi-\varpi_9)$")
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_ylabel(r"cluster centre $\Delta\varpi - 180^\circ$ [deg]")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.savefig(FIG / "fig9_nbody.png")
    plt.close(fig)


def fig_lsst_selection():
    path = CACHE / "S_bb16_lsst.npy"
    if not path.exists():
        print("  skip: LSST cube missing")
        return
    df = load_etnos("bb16")
    S_cur, S_l = selection_cube("bb16"), np.load(path)
    picks = [df.index[df["designation"] == d][0] for d in ("2012 VP113", "2013 SY99", "2015 GT50") if (df["designation"] == d).any()]
    fig, axes = plt.subplots(2, len(picks), figsize=(7.2, 4.4), sharex=True, sharey=True)
    for col, j in enumerate(picks):
        for row, (S, lab) in enumerate(((S_cur, "current surveys"), (S_l, "LSST model"))):
            ax = axes[row, col]
            n = S.shape[1]
            ax.imshow(S[j].T * n * n, origin="lower", extent=(0, 360, 0, 360), cmap="Greys", vmin=0, vmax=4,
                      aspect="auto")
            ax.plot(df["varpi"][j], df["Omega"][j], "o", color=ORANGE, ms=5)
            ax.grid(False)
            if row == 0:
                ax.set_title(df["designation"][j], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{lab}\n$\\Omega$ [deg]", fontsize=8)
            if row == 1:
                ax.set_xlabel(r"$\varpi$ [deg]")
    fig.savefig(FIG / "fig10_lsst_selection.png")
    plt.close(fig)


def fig_decision():
    if not _have("lsst_forecast"):
        return
    f = load("lsst_forecast")
    grid = f["grid"]
    fvals = sorted({r["f_max"] for r in grid})
    kvals = sorted({r["kappa_varpi"] for r in grid})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, sel in zip(axes, ("current", "lsst")):
        Z = np.array([[next(r["N_K100"] for r in grid if r["selection"] == sel and r["f_max"] == fv
                            and r["kappa_varpi"] == kv) for fv in fvals] for kv in kvals])
        Z = np.where(np.isfinite(Z), Z, 1e5)
        cs = ax.contourf(fvals, kvals, np.log10(Z), levels=np.linspace(1, 4, 13), cmap="viridis_r", extend="max")
        cl = ax.contour(fvals, kvals, Z, levels=[31, 62, 124, 250, 500, 1000], colors="white", linewidths=0.7)
        ax.clabel(cl, fmt="%d", fontsize=6.5)
        th = f["theta_implied"]
        ax.plot(th["f_max"], 10 ** th["log10_kvarpi"], "*", color=ORANGE, ms=10, label="implied by data")
        ts = f["theta_strong"]
        ax.plot(ts["f_max"], 10 ** ts["log10_kvarpi"], "D", color="white", mec=INK, ms=5, label="BB21-like")
        ax.set_yscale("log")
        ax.set_xlabel(r"confined fraction $f_\mathrm{max}$")
        ax.set_title({"current": "today's survey mix", "lsst": "LSST"}[sel], fontsize=9)
        ax.grid(False)
    axes[0].set_ylabel(r"perihelion concentration $\kappa_\varpi$")
    axes[0].legend(fontsize=6.5, loc="upper left")
    cb = fig.colorbar(cs, ax=axes, shrink=0.9)
    cb.set_label(r"$\log_{10} N$ for $K=100$")
    fig.savefig(FIG / "fig11_decision.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for fn in (fig_history, fig_predictive, fig_selmarg, fig_nbody, fig_lsst_selection, fig_decision):
        print(fn.__name__)
        fn()


if __name__ == "__main__":
    main()
