"""Run the statistics core on the canonical ETNO samples and write the numbers.

Replaces old scripts 04, 05, 08, 09, 12, 14, 21, 28 and 31. Every p-value is
either exact up to Monte Carlo error or calibrated by simulation; the uniform
null used here does NOT model survey selection (see planetix.bias for that),
so a small p-value here means "not uniform", not "planet".

Usage:
    python scripts/stats_report.py            # full run, a few minutes
    python scripts/stats_report.py --quick    # fewer simulations, for a smoke test
Writes results/stats_report.json and prints a summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from planetix.data import load_catalog, select_etnos
from planetix.stats import circular as C
from planetix.stats import mixtures as MX
from planetix.stats import multivariate as MV
from planetix.stats import nulls as N

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261030


def circular_block(df, n_sims, rng):
    out = {"n": len(df)}
    for col in ("varpi", "Omega", "omega"):
        r, k = C.rayleigh_test(df[col]), C.kuiper_test(df[col])
        ci = C.bootstrap_mean_direction(df[col], n_boot=5000, rng=rng)
        out[col] = {
            "R_bar": r.R_bar, "rayleigh_p": r.p_value,
            "rayleigh_p_mc": N.rayleigh_mc(df[col], n_sims=n_sims, rng=rng).p_value,
            "kuiper_V": k.V, "kuiper_p": k.p_value,
            "mean_dir": r.mean_direction, "mean_dir_ci95": [ci.lower, ci.upper],
            "kappa": C.kappa_mle(df[col]),
        }
    peri = C.perihelion_direction(df["i"], df["Omega"], df["omega"])
    sr = C.spherical_rayleigh(peri)
    out["perihelion_3d"] = {"R_bar": sr.R_bar, "p": sr.p_value}
    # mean orbit pole: descriptive only, a uniform-sphere null is meaningless for poles
    h = C.orbit_pole(df["i"], df["Omega"]).mean(axis=0)
    i_m, O_m = C.pole_to_elements(h)
    out["mean_pole"] = {"i": i_m, "Omega": O_m, "R_bar": float(np.linalg.norm(h))}
    corr = C.circular_correlation(df["varpi"], df["Omega"], n_perm=n_sims // 10, rng=rng)
    out["varpi_Omega_corr"] = {"r": corr.r, "p": corr.p_value}
    out["min_detectable_kappa_80pct"] = N.min_detectable_kappa(len(df), rng=rng)
    return out


def cut_scan(parent, n_sims, rng):
    a_grid, q_grid = (150, 200, 230, 250, 300), (30, 35, 40)
    masks, labels = [], []
    for a in a_grid:
        for q in q_grid:
            m = ((parent["a"] > a) & (parent["q"] > q)).to_numpy()
            if m.sum() >= 5:
                masks.append(m)
                labels.append(f"a>{a},q>{q}")
    r = N.scan_cuts(parent["varpi"].to_numpy(), masks, labels, n_sims=n_sims, rng=rng)
    return {
        "n_cuts": len(labels),
        "cuts": {l: {"n": int(n), "local_p": float(p)} for l, n, p in zip(labels, r.n_per_cut, r.local_p)},
        "best_cut": r.best_label, "best_local_p": r.best_local_p, "global_p": r.global_p,
    }


def mixture_block(theta, n_boot, rng):
    fits = {k: MX.fit_vonmises_mixture(theta, k, rng=rng) for k in (0, 1, 2)}
    out = {"bic": {k: f.bic for k, f in fits.items()},
           "fits": {k: {"weights": f.model.weights, "mus": f.model.mus, "kappas": f.model.kappas,
                        "w_uniform": f.model.w_uniform} for k, f in fits.items()}}
    out["lrt_0v1_p"] = MX.mixture_lrt(theta, 0, 1, n_boot=n_boot, n_init=5, rng=rng).p_value
    out["lrt_1v2_p"] = MX.mixture_lrt(theta, 1, 2, n_boot=n_boot, n_init=5, rng=rng).p_value
    for name, model in (("legacy_unimodal_p9", MX.LEGACY_UNIMODAL_P9), ("legacy_bimodal_p9", MX.LEGACY_BIMODAL_P9)):
        c = MX.compare_fixed_model(theta, model, n_sims=5000, rng=rng)
        out[name] = {"log10_BF_vs_uniform": c.log10_bayes_factor, "p_vs_uniform": c.p_value_vs_uniform}
    return out


def multivariate_block(df, n_null, rng):
    out = {}
    # old script 31: full orbital elements
    X, names = MV.orbital_features(df, ("a", "e", "i", "Omega", "omega"))
    pca = MV.pca_parallel_analysis(X, n_perm=n_null, rng=rng)
    out["pca"] = {"features": names, "evr": pca.explained_variance_ratio[:3].tolist(),
                  "null95": pca.null_95[:3].tolist(), "p": pca.p_values[:3].tolist(),
                  "n_significant": pca.n_significant,
                  "pc1_loadings": dict(zip(names, pca.components[0].round(3).tolist()))}
    sil = MV.calibrated_silhouette(X, n_null=n_null, rng=rng)
    out["silhouette_full"] = {"best_k": sil.best_k, "best": sil.best_silhouette, "p": sil.p_value,
                              "null_median": float(np.median(sil.null_max_silhouettes)),
                              "null_95": float(np.quantile(sil.null_max_silhouettes, 0.95))}
    # old scripts 21/28: (i, sin Omega, cos Omega) "warp" space
    W, _ = MV.orbital_features(df, ("i", "Omega"))
    sel = MV.gmm_select(W, rng=rng)
    out["warp_gmm"] = {"bic": sel.bic, "best_k_bic": sel.best_k,
                       "lrt_1v2_p": MV.gmm_bootstrap_lrt(W, 1, 2, n_boot=max(n_null // 2, 20), rng=rng).p_value}
    sw = MV.calibrated_silhouette(W, n_null=n_null, rng=rng)
    out["silhouette_warp"] = {"best_k": sw.best_k, "best": sw.best_silhouette, "p": sw.p_value}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_sims, n_boot, n_null = (5000, 49, 49) if args.quick else (100000, 299, 299)
    rng = np.random.default_rng(SEED)

    catalog = load_catalog()
    samples = {name: select_etnos(catalog, name) for name in ("ts14", "bb16", "detached")}
    report = {"seed": SEED, "n_sims": n_sims, "samples": {}}
    with threadpool_limits(1):
        for name, df in samples.items():
            print(f"[{name}] N={len(df)}")
            report["samples"][name] = {
                "circular": circular_block(df, n_sims, rng),
                "mixture_varpi": mixture_block(df["varpi"].to_numpy(), n_boot, rng),
                "multivariate": multivariate_block(df, n_null, rng),
            }
        print("[scan] look-elsewhere over a/q cuts")
        report["cut_scan"] = cut_scan(samples["ts14"], n_sims // 5, rng)

    out = ROOT / "results" / "stats_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
