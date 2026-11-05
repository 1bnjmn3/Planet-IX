"""Survey-bias analysis: clustering significance against bias-aware nulls, plus ML tests.

Usage:
    python scripts/run_bias.py            # full run, ~15 min on one core
    python scripts/run_bias.py --quick    # fewer simulations, for checking the pipeline

Writes results/bias/*.csv, results/bias/summary.json and, with matplotlib, figures.
Every random step is seeded, so a rerun gives the same numbers.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from planetix.bias import (
    BiasNull, EmpiricalFootprint, SurveySet, clustering_test, discovery_table, fit_hot_disk,
    inclination_selection, selection_grid,
)
from planetix.bias.orbits import geocentric, mean_motion
from planetix.bias.sky import galactic_latitude, galactic_plane_ecliptic
from planetix.data import CUTS, load_etnos
from planetix.ml import autoencoder_test, classifier_test

OUT = Path(__file__).resolve().parents[1] / "results" / "bias"
KEY_STATS = ["R_varpi", "D_varpi", "R_Omega", "R_perihelion", "D_perihelion", "R_pole", "min_p"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def geometry_validation(ss):
    """Predicted vs reported sky positions, and simulator recovery, for every characterised detection."""
    det = ss.detections()
    det = det[det["a"] > 0].reset_index(drop=True)
    M = mean_motion(det["a"]) * (det["JD"] - (det["tperi"] + 2400000.5))
    ra, dec, r, _ = geocentric(det["a"], det["e"], det["i"], det["Omega"], det["omega"], M, det["JD"], frame="equatorial")
    d1, d2 = np.radians(dec), np.radians(det["DEdeg"])
    cos_sep = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(np.radians(ra - det["RAdeg"]))
    det["sep_deg"] = np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))
    zero = {f: 0.0 for f in "gVrR"}  # Hsur is already in each survey's own filter
    det["p_detect"] = [
        ss.detection_probability(o.a, o.e, o.i, o.Hsur, [o.Omega], [o.omega], [m], epoch_jd=o.JD, colour=zero)[0]
        for o, m in zip(det.itertuples(), M)
    ]
    by = det.groupby("survey").agg(n=("p_detect", "size"), recovered=("p_detect", lambda x: (x > 0).mean()),
                                   median_p=("p_detect", "median"))
    return det, by


def cut_tests(disc, n_sims, rng):
    """Uniform vs empirical-footprint null for each named cut, with sensitivity variants."""
    variants = {
        "footprint (r=5, count)": {},
        "footprint (r=3)": {"radius": 3.0},
        "footprint (r=10)": {"radius": 10.0},
        "footprint (binary)": {"weighting": "binary"},
        "footprint (no Plutinos)": {"exclude_resonant": True},
    }
    rows, diags = [], []
    for cut in CUTS:
        s = load_etnos(cut)
        log(f"cut {cut}: N={len(s)}")
        res = clustering_test(s, BiasNull.uniform(s, rng=rng), n_sims, rng)
        rows += [{"cut": cut, "N": len(s), "null": "uniform", "stat": k, **res.loc[k].to_dict()} for k in res.index]
        for name, kw in variants.items():
            if cut != "bb16" and name != "footprint (r=5, count)":
                continue  # sensitivity variants only for the default sample
            fp = EmpiricalFootprint.from_catalog(exclude=s["designation"], discoveries=disc, **kw)
            null = BiasNull.from_model(s, fp, rng=rng)
            res = clustering_test(s, null, n_sims, rng)
            rows += [{"cut": cut, "N": len(s), "null": name, "stat": k, **res.loc[k].to_dict()} for k in res.index]
            if name == "footprint (r=5, count)":
                diags.append(null.diagnostics.assign(cut=cut))
    return pd.DataFrame(rows), pd.concat(diags, ignore_index=True)


def ossos_subset(ss, disc, n_sims, rng):
    """Shankman et al. (2017)-style test on the ETNOs the characterised surveys found."""
    det = ss.detections()
    c = CUTS["ts14"]
    sub = det[(det["a"] > c["a_min"]) & (det["q"] > c["q_min"])].copy()
    # H in the survey's filter; translate to V for the footprint (V - r = +0.28).
    filt_minus_v = {"OSSOS_v11": -0.28, "CFEPS_r": -0.28, "HiLat": -0.28, "MA": -0.28, "CFEPS_g": 0.42,
                    "CFEPS_presurvey": -0.38}
    sub["H"] = sub["Hsur"] - sub["survey"].map(filt_minus_v)
    sub["designation"] = sub["object"]
    sub = sub[["designation", "survey", "a", "e", "i", "Omega", "omega", "q", "H", "Hsur", "MPC"]].reset_index(drop=True)
    log(f"characterised-survey ETNOs (a>150, q>30): {len(sub)}")

    ossos_null = BiasNull.from_model(sub, ss, n_clones=100000, min_n_eff=300, max_clones=3000000, rng=rng)
    # The empirical footprint should not contain these objects either; match by MPC packed designation.
    excl = set(disc.loc[disc["packed"].isin(sub["MPC"]), "designation"])
    fp = EmpiricalFootprint.from_catalog(exclude=excl, discoveries=disc)
    fp_null = BiasNull.from_model(sub, fp, rng=rng)
    rows = []
    for name, null in [("uniform", BiasNull.uniform(sub, rng=rng)), ("OSSOS-characterised", ossos_null),
                       ("empirical footprint", fp_null)]:
        res = clustering_test(sub, null, n_sims, rng)
        rows += [{"null": name, "stat": k, **res.loc[k].to_dict()} for k in res.index]

    # Agreement of the two selection models, object by object: total-variation distance
    # between their S_j(varpi) marginals (0 = identical, 1 = disjoint), against the TV
    # distance between two independent estimates from the same model (sampling noise).
    agree = []
    for k, obj in sub.iterrows():
        h_o = np.histogram(np.mod(ossos_null.Omega_pool[k] + ossos_null.omega_pool[k], 360), 18, (0, 360))[0]
        h_f = np.histogram(np.mod(fp_null.Omega_pool[k] + fp_null.omega_pool[k], 360), 18, (0, 360))[0]
        h_u = np.histogram(rng.uniform(0, 360, len(fp_null.Omega_pool[k])), 18, (0, 360))[0]
        tv = lambda x, y: 0.5 * np.abs(x / x.sum() - y / y.sum()).sum()  # noqa: E731
        agree.append({"designation": obj["designation"], "a": obj["a"], "TV_ossos_vs_footprint": tv(h_o, h_f),
                      "TV_ossos_vs_uniform": tv(h_o, h_u), "TV_footprint_vs_uniform": tv(h_f, h_u),
                      "ossos_n_eff": ossos_null.diagnostics.loc[k, "n_eff"]})
    return sub, pd.DataFrame(rows), pd.DataFrame(agree)


def inclination_tests(fp, s, rng):
    P = inclination_selection(s, fp, rng=rng)
    biased, naive = fit_hot_disk(s["i"], P), fit_hot_disk(s["i"])
    # Old script 32: Anderson-Darling of the observed i against Rayleigh(10 deg) samples
    # passed through a box mask. Reproduced without the mask to show the dependence on the
    # fixed width: the conclusion flips with sigma, which the old script never varied.
    old = {}
    for sigma in (6.0, 10.0, 15.0, 20.0):
        null_i = rng.rayleigh(sigma, 10000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old[f"AD_p_rayleigh_{sigma:g}"] = float(stats.anderson_ksamp([s["i"].to_numpy(), null_i]).pvalue)
    return {"with_selection": {k: v for k, v in biased.items() if k not in ("loglik", "sigmas")},
            "no_selection": {k: v for k, v in naive.items() if k not in ("loglik", "sigmas")}, "old_script_32": old}


def old_autoencoder_audit(s, rng, n_seeds=10):
    """Re-run old script 02's recipe (with sklearn in place of TensorFlow) on the canonical sample.

    The old recipe: training set of 30,000 rows with a, e, i bootstrapped *independently*
    and varpi uniform; MinMax scaling of (a, e, i, sin varpi, cos varpi); 12-2-12
    autoencoder; real objects ranked by reconstruction error, no seed and no calibration.
    (Batch size 256 instead of 16, for speed; 60 epochs as in the original.)
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler

    def feats(a, e, i, varpi):
        t = np.radians(varpi)
        return np.column_stack([a, e, np.radians(i), np.sin(t), np.cos(t)])

    X_real = feats(s["a"], s["e"], s["i"], s["varpi"])
    tops, scores = [], []
    for seed in range(n_seeds):
        r = np.random.default_rng(seed)
        n = 30000
        X_syn = feats(r.choice(s["a"], n), r.choice(s["e"], n), r.choice(s["i"], n), r.uniform(0, 360, n))
        sc = MinMaxScaler().fit(X_syn)
        net = MLPRegressor(hidden_layer_sizes=(12, 2, 12), activation="relu", max_iter=60, batch_size=256, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.fit(sc.transform(X_syn), sc.transform(X_syn))
        Z = sc.transform(X_real)
        err = ((net.predict(Z) - Z) ** 2).mean(axis=1)
        scores.append(err)
        tops.append(set(s["designation"].to_numpy()[np.argsort(err)[::-1][:3]]))
    scores = np.array(scores)
    mean_score = scores.mean(axis=0)
    jacc = [len(x & y) / len(x | y) for k, x in enumerate(tops) for y in tops[k + 1:]]
    dv = np.abs((s["varpi"] - stats.circmean(s["varpi"], 360, 0) + 180) % 360 - 180)
    return {
        "top3_jaccard_between_seeds_mean": float(np.mean(jacc)),
        "distinct_objects_in_top3_over_seeds": len(set().union(*tops)),
        "spearman_score_vs_a": float(stats.spearmanr(mean_score, s["a"]).statistic),
        "spearman_score_vs_e": float(stats.spearmanr(mean_score, s["e"]).statistic),
        "spearman_score_vs_i": float(stats.spearmanr(mean_score, s["i"]).statistic),
        "spearman_score_vs_varpi_offset": float(stats.spearmanr(mean_score, dv).statistic),
        "spearman_score_vs_q": float(stats.spearmanr(mean_score, s["q"]).statistic),
    }


def old_milky_way_error():
    """Old script 17 drew the Galactic plane as 60 sin(lon - 280); compare with the real one."""
    lon, lat = galactic_plane_ecliptic()
    old = 60 * np.sin(np.radians(lon - 280))
    return {"max_abs_error_deg": float(np.max(np.abs(old - lat))), "true_max_lat": float(lat.max()),
            "true_lon_of_max": float(lon[np.argmax(lat)]), "old_lon_of_max": 10.0}


def figures(disc, fp, s, cut_df, sub, ossos_null_grids):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed; skipping figures")
        return
    from planetix.bias.sky import plot_sky

    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    sd = disc[disc["designation"].isin(s["designation"])]
    ax = plot_sky(sd, fp)
    ax.set_title("bb16 ETNOs at discovery vs. all other TNO discoveries (MPC snapshot)")
    ax.figure.savefig(figdir / "sky_discoveries.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    fig, axes = plt.subplots(1, len(ossos_null_grids), figsize=(3.2 * len(ossos_null_grids), 3), squeeze=False)
    for ax, (name, g_fp, g_os, obs) in zip(axes[0], ossos_null_grids):
        x = (np.arange(len(g_fp)) + 0.5) * 360 / len(g_fp)
        ax.step(x, g_fp * len(g_fp), where="mid", label="MPC footprint")
        ax.step(x, g_os * len(g_os), where="mid", label="OSSOS-characterised")
        ax.axvline(obs, c="k", ls=":", label="observed")
        ax.set(title=name, xlabel="varpi (deg)", xlim=(0, 360))
    axes[0, 0].set_ylabel("selection density / uniform")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "selection_varpi_ossos_vs_footprint.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_sims = 2000 if args.quick else 20000
    rng = np.random.default_rng(20261030)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    log("discovery table")
    disc = discovery_table()
    summary["discovery_dates"] = disc["disc_source"].value_counts().to_dict()

    log("characterised surveys: geometry validation")
    ss = SurveySet.load()
    summary["n_pointings"] = len(ss.pointings)
    det, by = geometry_validation(ss)
    by.to_csv(OUT / "simulator_recovery.csv")
    summary["position_error_deg"] = {f"p{q}": float(np.percentile(det["sep_deg"], q)) for q in (50, 90, 99, 100)}
    summary["simulator_recovery"] = {k: float(v) for k, v in by["recovered"].items()}
    summary["simulator_recovery_all"] = float((det["p_detect"] > 0).mean())

    log("clustering tests per cut")
    cut_df, diags = cut_tests(disc, n_sims, rng)
    cut_df.to_csv(OUT / "clustering_tests.csv", index=False)
    diags.to_csv(OUT / "footprint_selection_diagnostics.csv", index=False)
    summary["clustering_p"] = (cut_df[cut_df["stat"].isin(KEY_STATS)]
                               .pivot_table(index=["cut", "null"], columns="stat", values="p").round(4)
                               .reset_index().to_dict(orient="records"))

    log("OSSOS-characterised subset")
    sub, sub_df, agree = ossos_subset(ss, disc, n_sims, rng)
    sub.to_csv(OUT / "characterised_etnos.csv", index=False)
    sub_df.to_csv(OUT / "clustering_tests_characterised.csv", index=False)
    agree.to_csv(OUT / "selection_model_agreement.csv", index=False)
    summary["characterised_subset"] = {
        "N": len(sub),
        "p": sub_df[sub_df["stat"].isin(KEY_STATS)].pivot_table(index="null", columns="stat", values="p").round(4).to_dict(orient="index"),
        "median_TV": agree[["TV_ossos_vs_footprint", "TV_ossos_vs_uniform", "TV_footprint_vs_uniform"]].median().round(3).to_dict(),
    }

    s = load_etnos("bb16")
    fp = EmpiricalFootprint.from_catalog(exclude=s["designation"], discoveries=disc)
    log("inclination fit")
    summary["inclination_bb16"] = inclination_tests(fp, s, rng)

    log("ML tests")
    ml = []
    for cut in ("bb16", "ts14"):
        sc = load_etnos(cut)
        fpc = EmpiricalFootprint.from_catalog(exclude=sc["designation"], discoveries=disc)
        for name, null in [("uniform", BiasNull.uniform(sc, rng=rng)), ("footprint", BiasNull.from_model(sc, fpc, rng=rng))]:
            ae, per_obj = autoencoder_test(sc, null, n_calib=300 if args.quick else 1000, rng=rng)
            c2 = classifier_test(sc, null, n_perm=50 if args.quick else 300, rng=rng)
            ml.append({"cut": cut, "null": name, "autoencoder_p": ae["p"], "autoencoder_score": ae["score"],
                       "autoencoder_null_mean": ae["null_mean"], "c2st_auc": c2["auc"], "c2st_null_mean": c2["null_mean"],
                       "c2st_p": c2["p"]})
            if cut == "bb16" and name == "footprint":
                per_obj.to_csv(OUT / "autoencoder_per_object_bb16.csv", index=False)
                summary["autoencoder_objects_p_below_0.05"] = int((per_obj["p_object"] < 0.05).sum())
    pd.DataFrame(ml).to_csv(OUT / "ml_tests.csv", index=False)
    summary["ml"] = ml
    summary["old_script_02_audit"] = old_autoencoder_audit(s, rng)
    summary["old_script_17_milky_way"] = old_milky_way_error()
    sd = disc[disc["designation"].isin(s["designation"])]
    summary["bb16_discovered_within_10deg_of_galactic_plane"] = int((np.abs(galactic_latitude(sd["disc_lon"], sd["disc_lat"])) < 10).sum())
    summary["reference_discoveries_within_10deg_of_galactic_plane_frac"] = float(
        (np.abs(galactic_latitude(fp.reference["disc_lon"], fp.reference["disc_lat"])) < 10).mean())

    log("figures")
    grids = []
    ossos_like = sub[sub["a"] > 250]
    for _, obj in ossos_like.iterrows():
        excl = set(disc.loc[disc["packed"] == obj.get("MPC"), "designation"])
        fpo = EmpiricalFootprint.from_catalog(exclude=excl, discoveries=disc)
        g_fp = selection_grid(obj, fpo, bins=(36, 1), rng=rng)[:, 0]
        g_os = selection_grid(obj, ss, bins=(36, 1), n_clones=400000, rng=rng)[:, 0]
        grids.append((obj["designation"], g_fp, g_os, (obj["Omega"] + obj["omega"]) % 360))
    figures(disc, fp, s, cut_df, sub, grids)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done: {OUT}")


if __name__ == "__main__":
    main()
