"""Bayes factor for a distant planet against no planet, with every robustness run.

Writes JSON results and posterior samples to paper/results/. Stages:

    main        baseline evidence and posterior for each sample cut
    priors      prior-sensitivity runs on bb16
    selection   no selection correction, and other footprint radii
    split       objects discovered before 2016 against the whole sample
    loo         leave-one-out influence of each object (bb16)
    nondet      as main, also requiring that direct searches missed the planet
    calib       simulated no-planet and planet catalogues (false-positive rate and power)
    calib2      no-planet catalogues for ts14, and with a mis-specified selection function

Usage:
    python scripts/infer_planet9.py                 # all stages
    python scripts/infer_planet9.py main priors     # chosen stages
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from planetix.data import load_etnos
from planetix.data.catalog import CUTS, DEFAULT_SNAPSHOT
from planetix.data.mpc import read_mpcorb
from planetix.inference.evidence import draw_orientations, planet_density_grid, run_evidence
from planetix.inference.model import PARAMS, Likelihood, Prior
from planetix.inference.nondetection import NonDetection
from planetix.inference.selection import Footprint, discovery_jd, selection_grid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "results"
CACHE = ROOT / "paper" / "cache"
NLIVE = 500
NLIVE_SIM = 250

# Reference parameters for injection tests: Brown & Batygin (2021, AJ 162, 219) best fit,
# with a confined population as strong as our bb16 posterior median.
THETA_INJECT = [np.log10(6.2), np.log10(380.0), 0.2, 16.0, 247.0, 96.0, 0.6, np.log10(4.0), np.log10(20.0)]


def _footprint(radius_deg=2.0, a_min=0.0):
    # Survey points exclude every ETNO in the widest cut, so the objects whose
    # clustering is being tested cannot shape their own selection function.
    exclude = load_etnos("ts14")["designation"]
    return Footprint.from_catalog(radius_deg=radius_deg, exclude=exclude, a_min=a_min)


def selection_cube(cut: str, radius_deg: float = 2.0, a_min: float = 0.0) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (f"S_{cut}_r{radius_deg:g}" + (f"_amin{a_min:g}" if a_min else "") + ".npy")
    if path.exists():
        return np.load(path)
    fp = _footprint(radius_deg, a_min)
    S = np.stack([selection_grid(o, fp) for _, o in load_etnos(cut).iterrows()])
    np.save(path, S)
    return S


def discovery_years(df):
    cat = read_mpcorb(DEFAULT_SNAPSHOT).set_index("designation")
    jd = discovery_jd(cat.loc[df["designation"]].reset_index())
    return np.floor(2000.0 + (jd - 2451545.0) / 365.25).astype(int)


def summarise(res, prior: Prior | None = None, **extra):
    q = np.percentile(res.samples, [2.5, 16, 50, 84, 97.5], axis=0)
    return {
        "ln_bf": res.log_bf, "ln_bf_err": res.log_bf_err, "bf": res.bf,
        "p_planet_even_odds": res.p_planet(1.0), "n_obj": res.n_obj,
        "posterior": {p: dict(zip(["p2.5", "p16", "p50", "p84", "p97.5"], q[:, k].tolist())) for k, p in enumerate(PARAMS)},
        "prior": {p: list(getattr(prior or Prior(), p)) for p in PARAMS},
        **extra,
    }


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"  -> {name}.json")


def stage_main():
    out = {}
    for cut in CUTS:
        df = load_etnos(cut)
        like = Likelihood.from_frame(df, selection_cube(cut))
        res = run_evidence(like, nlive=NLIVE)
        np.save(OUT / f"posterior_{cut}.npy", res.samples)
        out[cut] = summarise(res, cut=cut, cut_def=CUTS[cut])
        print(f"  {cut}: N={len(df)} lnBF={res.log_bf:.2f}±{res.log_bf_err:.2f}")
        # per-object contribution at the posterior median
        if cut == "bb16":
            med = np.median(res.samples, axis=0)
            out[cut]["per_object_at_median"] = dict(zip(df["designation"], like.per_object(med).tolist()))
            out[cut]["confined_fraction_at_median"] = dict(zip(df["designation"], like.confined_fraction(med).tolist()))
    _save("main", out)


def stage_nondet():
    out = {}
    for cut in CUTS:
        df = load_etnos(cut)
        like = Likelihood.from_frame(df, selection_cube(cut), nondetection=NonDetection())
        res = run_evidence(like, nlive=NLIVE)
        np.save(OUT / f"posterior_{cut}_nondet.npy", res.samples)
        out[cut] = summarise(res, cut=cut)
        print(f"  {cut}: lnBF={res.log_bf:.2f}")
    # the same term alone: prior-averaged probability that the searches missed the planet
    rng = np.random.default_rng(0)
    nd = NonDetection()
    draws = np.array([Prior().transform(u) for u in rng.random((4000, len(PARAMS)))])
    out["prior_mean_p_missed"] = float(np.mean([np.exp(nd.log_p_missed(t)) for t in draws]))
    _save("nondet", out)


PRIOR_VARIANTS = {
    "baseline": {},
    "wide_mass_a": {"log10_m9": (np.log10(0.5), np.log10(30.0)), "log10_a9": (np.log10(200.0), np.log10(1500.0))},
    "narrow_bb21": {"log10_m9": (np.log10(4.0), np.log10(10.0)), "log10_a9": (np.log10(300.0), np.log10(600.0)), "i9": (0.0, 30.0)},
    "wide_kappa": {"log10_kvarpi": (-0.5, np.log10(50.0)), "log10_kpole": (0.0, np.log10(300.0))},
    "tight_kappa": {"log10_kvarpi": (np.log10(2.0), np.log10(10.0)), "log10_kpole": (1.0, 2.0)},
    "fmax_le_0.5": {"f_max": (0.0, 0.5)},
    "fmax_ge_0.5": {"f_max": (0.5, 1.0)},
    "low_incl": {"i9": (0.0, 25.0)},
    "wide_incl": {"i9": (0.0, 90.0)},
}


def stage_priors():
    df = load_etnos("bb16")
    S = selection_cube("bb16")
    out = {}
    for name, kw in PRIOR_VARIANTS.items():
        prior = replace(Prior(), **kw)
        res = run_evidence(Likelihood.from_frame(df, S), prior, nlive=NLIVE)
        out[name] = summarise(res, prior)
        print(f"  prior {name}: lnBF={res.log_bf:.2f}")
    for w in (0.1, 0.4):
        res = run_evidence(Likelihood.from_frame(df, S, w=w), nlive=NLIVE)
        out[f"width_w{w}"] = summarise(res, w=w)
        print(f"  width w={w}: lnBF={res.log_bf:.2f}")
    _save("priors", out)


def stage_selection():
    df = load_etnos("bb16")
    out = {}
    flat = np.full(selection_cube("bb16").shape, 1.0 / np.prod(selection_cube("bb16").shape[1:]))
    res = run_evidence(Likelihood.from_frame(df, flat), nlive=NLIVE)
    out["no_selection"] = summarise(res)
    print(f"  no selection: lnBF={res.log_bf:.2f}")
    for r in (1.0, 4.0):
        res = run_evidence(Likelihood.from_frame(df, selection_cube("bb16", r)), nlive=NLIVE)
        out[f"radius_{r:g}deg"] = summarise(res, radius_deg=r)
        print(f"  radius {r}: lnBF={res.log_bf:.2f}")
    # survey points only from scattered/detached objects (a > 50 AU), radius 4 deg to
    # make up for the sparser points
    for cut in ("bb16", "ts14"):
        res = run_evidence(Likelihood.from_frame(load_etnos(cut), selection_cube(cut, 4.0, 50.0)), nlive=NLIVE)
        out[f"{cut}_footprint_a_gt_50"] = summarise(res, radius_deg=4.0, a_min=50.0)
        print(f"  {cut} footprint a>50: lnBF={res.log_bf:.2f}")
    _save("selection", out)


def stage_split():
    df = load_etnos("bb16")
    S = selection_cube("bb16")
    years = discovery_years(df)
    early = years < 2016
    out = {"discovery_year": dict(zip(df["designation"], years.tolist()))}
    for name, m in (("before_2016", early), ("all", np.ones(len(df), bool))):
        res = run_evidence(Likelihood.from_frame(df[m].reset_index(drop=True), S[m]), nlive=NLIVE)
        out[name] = summarise(res)
        print(f"  {name}: N={m.sum()} lnBF={res.log_bf:.2f}")
    out["new_objects_conditional_ln_bf"] = out["all"]["ln_bf"] - out["before_2016"]["ln_bf"]
    out["n_new"] = int((~early).sum())
    _save("split", out)


def _loo(args):
    k, = args
    df = load_etnos("bb16")
    S = selection_cube("bb16")
    m = np.arange(len(df)) != k
    res = run_evidence(Likelihood.from_frame(df[m].reset_index(drop=True), S[m]), nlive=NLIVE_SIM, seed=k)
    return df["designation"][k], res.log_bf


def stage_loo():
    n = len(load_etnos("bb16"))
    with Pool() as pool:
        out = dict(pool.map(_loo, [(k,) for k in range(n)]))
    _save("loo", out)


def _sim(args):
    kind, n_obj, seed = args
    df = load_etnos("bb16")
    S = selection_cube("bb16")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), n_obj, replace=n_obj > len(df)) if n_obj != len(df) else np.arange(len(df))
    sub = df.iloc[idx].reset_index(drop=True)
    S_sub = S[idx]
    g = None
    if kind == "planet":
        g = planet_density_grid(Likelihood.from_frame(sub, S_sub), THETA_INJECT)
    sub = sub.copy()
    sub["varpi"], sub["Omega"] = draw_orientations(S_sub, rng, g)
    res = run_evidence(Likelihood.from_frame(sub, S_sub), nlive=NLIVE_SIM, seed=seed)
    return {"kind": kind, "n_obj": n_obj, "seed": seed, "ln_bf": res.log_bf}


def stage_calib(n_null=200, n_planet=100, sizes=(31, 62, 124)):
    jobs = [("null", 31, s) for s in range(n_null)]
    for n in sizes:
        jobs += [("planet", n, 10_000 + 1000 * n + s) for s in range(n_planet)]
        if n != 31:
            jobs += [("null", n, 20_000 + 1000 * n + s) for s in range(n_planet // 2)]
    with Pool() as pool:
        out = pool.map(_sim, jobs, chunksize=2)
    _save("calibration", {"theta_inject": dict(zip(PARAMS, THETA_INJECT)), "runs": out})


def _sim_cross(args):
    cut, seed, cross = args
    df = load_etnos(cut)
    S = selection_cube(cut)
    # cross: the catalogue is drawn through a different selection function than the one analysed
    S_true = selection_cube(cut, 4.0, 50.0) if cross else S
    rng = np.random.default_rng(seed)
    sub = df.copy()
    sub["varpi"], sub["Omega"] = draw_orientations(S_true, rng)
    res = run_evidence(Likelihood.from_frame(sub, S), nlive=NLIVE_SIM, seed=seed)
    return {"cut": cut, "cross": cross, "seed": seed, "ln_bf": res.log_bf}


def stage_calib2(n=100):
    """No-planet catalogues for ts14, and catalogues drawn through the alternative
    footprint but analysed with the baseline one (selection-function error)."""
    jobs = [("ts14", 30_000 + s, False) for s in range(n)]
    jobs += [(cut, 40_000 + s, True) for cut in ("bb16", "ts14") for s in range(n)]
    with Pool() as pool:
        out = pool.map(_sim_cross, jobs, chunksize=2)
    _save("calibration2", {"runs": out})


STAGES = {"main": stage_main, "priors": stage_priors, "selection": stage_selection,
          "split": stage_split, "loo": stage_loo, "nondet": stage_nondet, "calib": stage_calib,
          "calib2": stage_calib2}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for cut in CUTS:  # build selection grids once, before any worker pool forks
        selection_cube(cut)
    for name in args.stages:
        t = time.time()
        print(f"[{name}]")
        STAGES[name]()
        print(f"  {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
