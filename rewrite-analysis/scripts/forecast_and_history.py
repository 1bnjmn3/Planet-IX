"""Evidence history, out-of-sample test, selection-marginalised evidence and LSST forecast.

Builds on scripts/infer_planet9.py (same likelihood, priors and selection cubes)
and writes paper/results/<stage>.json. Stages:

    history     ln K for the cumulative bb16 sample, year by year of discovery
    predictive  the planet inferred from pre-2016 objects, tested on later ones,
                calibrated against posterior-predictive and no-planet catalogues
    selmarg     ln K averaged over selection models (discrete set, and a continuous
                interpolation between the two most different footprints), each
                weighted by how well it predicts the observed orientations under H0
    lsst        LSST selection functions, information rates and the number of
                ETNOs needed to reach K = 10 / 100, for a grid of planet strengths
    lsstsim     simulated LSST catalogues analysed end to end (checks the forecast)
    yield       OSSOS-calibrated number of a > 250 AU ETNOs LSST should find

Usage: python scripts/forecast_and_history.py [stage ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_planet9 import CACHE, NLIVE, NLIVE_SIM, OUT, THETA_INJECT, selection_cube, summarise  # noqa: E402

from planetix.data import load_etnos  # noqa: E402
from planetix.inference.evidence import draw_orientations, planet_density_grid, run_evidence  # noqa: E402
from planetix.inference.lsst import LSSTFootprint, information_rate, n_required  # noqa: E402
from planetix.inference.model import PARAMS, Likelihood, Prior  # noqa: E402
from planetix.inference.selection import selection_grid  # noqa: E402

# Provisional-designation year for numbered or named objects whose catalogue
# name is not a provisional designation (MPC discovery circumstances).
DISCOVERY_YEAR = {"Sedna": 2003, "Alicanto": 2004, "Leleakuhonua": 2015}


def discovery_year(df):
    yr = df["name"].str.extract(r"^(\d{4}) [A-Z]{2}", expand=False).astype(float)
    return yr.fillna(df["name"].map(DISCOVERY_YEAR)).astype(int).to_numpy()


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"  -> {name}.json")


# ---------------------------------------------------------------- history
def _hist_job(args):
    year, cut = args
    df = load_etnos(cut)
    S = selection_cube(cut)
    m = discovery_year(df) <= year
    res = run_evidence(Likelihood.from_frame(df[m].reset_index(drop=True), S[m]), nlive=NLIVE, seed=year)
    np.save(CACHE / f"post_hist_{cut}_{year}.npy", res.samples)
    return cut, year, int(m.sum()), res.log_bf, res.log_bf_err


def stage_history(n_proc=1):
    jobs = []
    for cut in ("bb16", "ts14"):
        years = sorted(set(discovery_year(load_etnos(cut))))
        jobs += [(y, cut) for y in years if (discovery_year(load_etnos(cut)) <= y).sum() >= 3]
    with Pool(n_proc) as pool:
        rows = pool.map(_hist_job, jobs)
    out = {cut: [dict(year=y, n=n, ln_bf=l, ln_bf_err=e) for c, y, n, l, e in rows if c == cut] for cut in ("bb16", "ts14")}
    df = load_etnos("bb16")
    out["bb16_objects"] = dict(zip(df["designation"], discovery_year(df).tolist()))
    _save("history", out)
    for r in out["bb16"]:
        print(f"  bb16 <= {r['year']}: N={r['n']} lnK={r['ln_bf']:.2f}")


# ---------------------------------------------------------------- predictive
def _log_ratio_matrix(like: Likelihood, thetas, varpi, Omega):
    """ln[p(x_j | theta, H1) / p(x_j | H0)] for each theta (rows) and object (cols)."""
    out = np.empty((len(thetas), len(varpi)))
    keep = (like.varpi.copy(), like.Omega.copy())
    like.varpi, like.Omega = np.asarray(varpi, float), np.asarray(Omega, float)
    for k, th in enumerate(thetas):
        out[k] = like.per_object(th)
    like.varpi, like.Omega = keep
    return out


def _cond_lnk(M):
    """ln of the posterior-predictive Bayes factor: log mean_theta exp(sum_j M[theta, j])."""
    s = M.sum(axis=1)
    return float(np.log(np.mean(np.exp(s - s.max()))) + s.max())


def stage_predictive(split_year=2015, n_sim=1000, n_theta=1500, seed=7, radius=2.0, a_min=0.0):
    """Train on objects discovered up to split_year, test on the rest.

    The test statistic is the conditional Bayes factor of the new objects,
    K_new = p(x_new | x_old, H1) / p(x_new | H0), estimated with the training
    posterior. It is compared with its distribution when the new orientations
    are drawn (a) from the posterior predictive of the training data (the planet
    the old objects imply is real), (b) from the no-planet hypothesis.
    """
    rng = np.random.default_rng(seed)
    df = load_etnos("bb16")
    S = selection_cube("bb16", radius, a_min)
    tag = "" if (radius, a_min) == (2.0, 0.0) else f"_r{radius:g}_amin{a_min:g}"
    yr = discovery_year(df)
    tr, te = yr <= split_year, yr > split_year
    post_path = CACHE / f"post_hist_bb16_{split_year}{tag}.npy"
    if post_path.exists():
        post = np.load(post_path)
    else:
        post = run_evidence(Likelihood.from_frame(df[tr].reset_index(drop=True), S[tr]), nlive=NLIVE).samples
        np.save(post_path, post)
    thetas = post[rng.choice(len(post), min(n_theta, len(post)), replace=False)]
    test = df[te].reset_index(drop=True)
    like = Likelihood.from_frame(test, S[te])
    obs = _cond_lnk(_log_ratio_matrix(like, thetas, test["varpi"], test["Omega"]))
    per_obj = {d: _cond_lnk(_log_ratio_matrix(like, thetas, test["varpi"], test["Omega"])[:, [j]])
               for j, d in enumerate(test["designation"])}

    # The denominator of per_object does not depend on the data: reuse it across simulations.
    sims = {"planet": [], "null": []}
    for s in range(n_sim):
        th = post[rng.integers(len(post))]
        v, o = draw_orientations(like.S, rng, planet_density_grid(like, th))
        sims["planet"].append(_cond_lnk(_log_ratio_matrix(like, thetas[:400], v, o)))
        v, o = draw_orientations(like.S, rng)
        sims["null"].append(_cond_lnk(_log_ratio_matrix(like, thetas[:400], v, o)))
    pl, nu = np.array(sims["planet"]), np.array(sims["null"])
    out = {"split_year": split_year, "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "test_objects": test["designation"].tolist(), "ln_k_new_observed": obs,
           "ln_k_new_per_object": per_obj,
           "p_value_under_trained_planet": float(np.mean(pl <= obs)),
           "p_value_under_no_planet": float(np.mean(nu >= obs)),
           "frac_null_below_obs": float(np.mean(nu <= obs)),
           "planet_quantiles": np.percentile(pl, [5, 16, 50, 84, 95]).tolist(),
           "null_quantiles": np.percentile(nu, [5, 16, 50, 84, 95]).tolist(),
           # Bayes factor of "no planet" vs "the trained planet" from the new data alone is 1 / K_new
           "sims_planet": pl.tolist(), "sims_null": nu.tolist()}
    out["selection"] = {"radius_deg": radius, "a_min": a_min}
    _save(f"predictive_{split_year}{tag}", out)
    print(f"  train<= {split_year} (N={tr.sum()}), test N={te.sum()}: lnK_new={obs:.2f}; "
          f"P(<=obs | trained planet)={out['p_value_under_trained_planet']:.3f}; "
          f"P(>=obs | no planet)={out['p_value_under_no_planet']:.3f}")


# ---------------------------------------------------------------- selection marginalisation
def _cell_index(varpi, Omega, shape):
    nv, no = shape
    return (np.asarray(varpi) // (360.0 / nv)).astype(int) % nv, (np.asarray(Omega) // (360.0 / no)).astype(int) % no


def ln_z0(df, S):
    """ln p(orientations | H0, S) relative to an isotropic, unbiased sky (sum over objects)."""
    iv, io = _cell_index(df["varpi"], df["Omega"], S.shape[1:])
    n_cells = S.shape[1] * S.shape[2]
    return float(np.sum(np.log(S[np.arange(len(df)), iv, io] * n_cells)))


def _selmarg_job(args):
    name, S = args
    df = load_etnos("bb16")
    res = run_evidence(Likelihood.from_frame(df, S), nlive=NLIVE)
    return name, res.log_bf, res.log_bf_err


def _geo_mix(S_a, S_b, alpha):
    with np.errstate(divide="ignore"):
        L = (1 - alpha) * np.log(np.maximum(S_a, 1e-300)) + alpha * np.log(np.maximum(S_b, 1e-300))
    S = np.exp(L - L.max(axis=(1, 2), keepdims=True))
    return S / S.sum(axis=(1, 2), keepdims=True)


def stage_selmarg(n_proc=1):
    df = load_etnos("bb16")
    models = {
        "r1": selection_cube("bb16", 1.0), "r2": selection_cube("bb16", 2.0), "r4": selection_cube("bb16", 4.0),
        "r2_amin50": selection_cube("bb16", 2.0, 50.0), "r4_amin50": selection_cube("bb16", 4.0, 50.0),
    }
    bias_fp = CACHE / "S_bb16_biasfp.npy"
    if bias_fp.exists():
        models["bias_footprint"] = np.load(bias_fp)
    alphas = np.linspace(0, 1, 11)
    cont = {f"alpha{a:.1f}": _geo_mix(models["r2"], models["r4_amin50"], a) for a in alphas}
    jobs = list(models.items()) + list(cont.items())
    with Pool(n_proc) as pool:
        res = {n: (l, e) for n, l, e in pool.map(_selmarg_job, jobs)}
    allS = {**models, **cont}
    rows = {n: {"ln_bf": res[n][0], "ln_bf_err": res[n][1], "ln_z0": ln_z0(df, allS[n])} for n in allS}

    def bma(names):
        z0 = np.array([rows[n]["ln_z0"] for n in names])
        lk = np.array([rows[n]["ln_bf"] for n in names])
        w = np.exp(z0 - z0.max())
        w /= w.sum()
        return {"ln_bf": float(np.log(np.sum(w * np.exp(lk)))), "weights": dict(zip(names, w.tolist()))}

    out = {"models": rows, "bma_discrete": bma(list(models)), "bma_continuous": bma(list(cont)),
           "alphas": alphas.tolist()}
    _save("selmarg", out)
    for n, r in rows.items():
        print(f"  {n}: lnK={r['ln_bf']:.2f} lnZ0={r['ln_z0']:.2f}")
    print("  BMA discrete", out["bma_discrete"]["ln_bf"], " continuous", out["bma_continuous"]["ln_bf"])


# ---------------------------------------------------------------- LSST
def lsst_cube(cut="bb16", fp: LSSTFootprint | None = None, tag="") -> np.ndarray:
    path = CACHE / f"S_{cut}_lsst{tag}.npy"
    if path.exists():
        return np.load(path)
    fp = fp or LSSTFootprint()
    S = np.stack([selection_grid(o, fp) for _, o in load_etnos(cut).iterrows()])
    np.save(path, S)
    return S


def _fit_occam(power_runs, D_mean):
    """Fit median ln K(N) = N D - c ln N - c0 to simulated catalogues at known D."""
    runs = [r for r in power_runs if r["kind"] == "planet"]
    Ns = sorted({r["n_obj"] for r in runs})
    med = np.array([np.median([r["ln_bf"] for r in runs if r["n_obj"] == n]) for n in Ns])
    A = np.stack([-np.log(Ns), -np.ones(len(Ns))], axis=1)
    c, c0 = np.linalg.lstsq(A, med - D_mean * np.array(Ns), rcond=None)[0]
    # scatter of ln K grows like s sqrt(N): half the 16-84% width
    half = np.array([np.diff(np.percentile([r["ln_bf"] for r in runs if r["n_obj"] == n], [16, 84]))[0] / 2 for n in Ns])
    s = float(np.sum(half * np.sqrt(Ns)) / np.sum(Ns))
    return float(c), float(c0), Ns, med.tolist(), s


def n_assured(D, ln_k_target, c, c0, s, z=1.2816, n_max=100_000):
    """Smallest N at which 90% (z = 1.28) of catalogues reach the target, Gaussian scatter s sqrt(N)."""
    n = np.arange(max(10, int(np.ceil(c / max(D, 1e-12)))), n_max + 1)
    ok = np.flatnonzero(n * D - c * np.log(n) - c0 - z * s * np.sqrt(n) >= ln_k_target)
    return float(n[ok[0]]) if len(ok) else np.inf


def stage_lsst():
    df = load_etnos("bb16")
    S_cur = selection_cube("bb16")
    S_lsst = lsst_cube()
    S_lsst_deep = lsst_cube(fp=LSSTFootprint(v_lim=25.5), tag="_deep")
    power = json.loads((OUT / "power.json").read_text())
    theta_impl = [power["theta_inject"][p] for p in PARAMS]
    out = {"theta_implied": dict(zip(PARAMS, theta_impl)), "theta_strong": dict(zip(PARAMS, THETA_INJECT))}

    # Occam terms calibrated on the existing injections at the implied planet (current selection)
    D_cur = information_rate(Likelihood.from_frame(df, S_cur), theta_impl)
    c, c0, Ns, med, s = _fit_occam(power["runs"], float(D_cur.mean()))
    out["occam_fit"] = {"c": c, "c0": c0, "s": s, "N": Ns, "median_ln_bf": med, "D_mean_current": float(D_cur.mean())}

    # information rate for each selection model, at a grid of planet strengths
    fgrid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    kgrid = [1.5, 2.0, 2.7, 4.0, 6.0, 10.0]
    rows = []
    for sel_name, S in (("current", S_cur), ("lsst", S_lsst), ("lsst_deep", S_lsst_deep)):
        like = Likelihood.from_frame(df, S)
        for f in fgrid:
            for k in kgrid:
                th = list(theta_impl)
                th[PARAMS.index("f_max")] = f
                th[PARAMS.index("log10_kvarpi")] = np.log10(k)
                D = float(information_rate(like, th).mean())
                rows.append({"selection": sel_name, "f_max": f, "kappa_varpi": k, "D": D,
                             "N_K10": n_required(D, np.log(10), c, c0), "N_K100": n_required(D, np.log(100), c, c0),
                             "N90_K100": n_assured(D, np.log(100), c, c0, s)})
        for name, th in (("implied", theta_impl), ("strong", THETA_INJECT)):
            D = float(information_rate(like, th).mean())
            out[f"{sel_name}_{name}"] = {"D": D, "N_K10": n_required(D, np.log(10), c, c0),
                                         "N_K100": n_required(D, np.log(100), c, c0),
                                         "N90_K100": n_assured(D, np.log(100), c, c0, s)}
            r = out[f"{sel_name}_{name}"]
            print(f"  {sel_name} {name}: D={D:.3f} N(K=10)={r['N_K10']:.0f} N(K=100)={r['N_K100']:.0f} "
                  f"N90(K=100)={r['N90_K100']:.0f}")
    # how much of the orientation space each selection model sees (1 = uniform)
    for sel_name, S in (("current", S_cur), ("lsst", S_lsst), ("lsst_deep", S_lsst_deep)):
        n_cells = S.shape[1] * S.shape[2]
        neff = 1.0 / np.sum(S**2, axis=(1, 2)) / n_cells
        out[f"{sel_name}_coverage_median"] = float(np.median(neff))
    out["grid"] = rows
    _save("lsst_forecast", out)


def _lsst_sim(args):
    n_obj, seed, kind = args
    df = load_etnos("bb16")
    S = lsst_cube()
    power = json.loads((OUT / "power.json").read_text())
    theta = [power["theta_inject"][p] for p in PARAMS]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), n_obj, replace=True)
    sub, S_sub = df.iloc[idx].reset_index(drop=True).copy(), S[idx]
    g = planet_density_grid(Likelihood.from_frame(sub, S_sub), theta) if kind == "planet" else None
    sub["varpi"], sub["Omega"] = draw_orientations(S_sub, rng, g)
    res = run_evidence(Likelihood.from_frame(sub, S_sub), nlive=NLIVE_SIM, seed=seed)
    return {"n_obj": n_obj, "seed": seed, "kind": kind, "ln_bf": res.log_bf}


def stage_lsstsim(n=16, sizes=(31, 62, 124), n_proc=1):
    jobs = [(m, 70_000 + 1000 * m + s, "planet") for m in sizes for s in range(n)]
    jobs += [(124, 80_000 + s, "null") for s in range(n // 2)]
    with Pool(n_proc) as pool:
        runs = pool.map(_lsst_sim, jobs, chunksize=1)
    _save("lsst_sims", {"runs": runs})
    for m in sizes:
        v = [r["ln_bf"] for r in runs if r["n_obj"] == m and r["kind"] == "planet"]
        print(f"  LSST N={m}: median lnK={np.median(v):.2f}")


# ---------------------------------------------------------------- yield
def _population(n, rng, slope_bright=0.9, slope_faint=0.5, H_break=8.3, H_min=3.0, H_max=10.5, a_slope=-0.5):
    """Model a > 250 AU, q > 30 AU population: dN/da ~ a^a_slope (250-1500 AU), q uniform 30-50 AU,
    i ~ sin i x N(0, 15 deg), isotropic angles, broken power-law H_r."""
    u = rng.random(n)
    lo, hi = 250.0, 1500.0
    p = a_slope + 1
    a = (lo**p + u * (hi**p - lo**p)) ** (1 / p)
    q = rng.uniform(30, 50, n)
    inc = np.abs(rng.normal(0, 15, 6 * n))
    inc = inc[rng.random(inc.size) < np.sin(np.radians(inc))][:n]
    # broken power law in H: dN/dH ~ 10^(slope H), continuous at the break
    Hs = np.linspace(H_min, H_max, 2000)
    dens = np.where(Hs < H_break, 10 ** (slope_bright * (Hs - H_break)), 10 ** (slope_faint * (Hs - H_break)))
    cdf = np.cumsum(dens)
    H = np.interp(rng.random(n) * cdf[-1], cdf, Hs)
    ang = rng.uniform(0, 360, (3, n))
    return a, 1 - q / a, inc, H, ang


def stage_yield(n_mc=1_500_000, seed=11):
    """N_LSST = N_OSSOS,obs x <P_LSST> / <P_OSSOS> for a model population, with the
    OSSOS-family characterised surveys (Bannister et al. 2018) as the anchor."""
    from scipy.stats import gamma

    from planetix.bias import SurveySet
    from planetix.inference.kepler import apparent_mag, sky_position

    rng = np.random.default_rng(seed)
    ss = SurveySet.load()
    det = ss.detections()
    n_obs = int(((det["a"] > 250) & (det["q"] > 30)).sum())
    out = {"n_ossos_observed": n_obs, "variants": {}}
    variants = {
        "baseline": {}, "steep_H": {"slope_faint": 0.7}, "shallow_H": {"slope_faint": 0.3},
        "flat_a": {"a_slope": 0.0}, "steep_a": {"a_slope": -1.5},
    }
    lo, hi = gamma.ppf([0.16, 0.84], n_obs + 0.5)  # Jeffreys interval on the Poisson mean
    n_orbit, n_ang = n_mc // 1000, 1000
    for name, kw in variants.items():
        a, e, inc, H_r, _ = _population(n_orbit, rng, **kw)
        H_V = H_r + 0.5  # V - r for red TNOs
        p_ossos, lon, lat, r, HH = [], [], [], [], []
        for k in range(n_orbit):  # detection_probability takes one (a, e, i, H) with many orientations
            Om, om, M = rng.uniform(0, 360, (3, n_ang))
            p_ossos.append(ss.detection_probability(a[k], e[k], inc[k], H_V[k], Om, om, M))
            lo_, la_, r_ = sky_position(a[k], e[k], inc[k], Om, om, M)
            lon.append(lo_), lat.append(la_), r.append(r_), HH.append(np.full(n_ang, H_V[k]))
        p_ossos, lon, lat, r, HH = (np.concatenate(x) for x in (p_ossos, lon, lat, r, HH))
        out_v = {}
        for vlim, rmax in ((24.0, np.inf), (24.5, np.inf), (25.0, np.inf), (24.5, 150.0), (24.5, 100.0)):
            fp = LSSTFootprint(v_lim=vlim)
            p_lsst = fp.weight(lon, lat, apparent_mag(HH, r)) * (r < rmax)
            ratio = p_lsst.mean() / p_ossos.mean()
            out_v[f"V{vlim}" + (f"_r<{rmax:g}" if np.isfinite(rmax) else "")] = {"N_lsst": n_obs * ratio, "N_lsst_16": lo * ratio, "N_lsst_84": hi * ratio,
                                 "P_lsst": float(p_lsst.mean()), "P_ossos": float(p_ossos.mean())}
        out["variants"][name] = out_v
        print(f"  {name}: " + ", ".join(f"{k}: {v['N_lsst']:.0f} ({v['N_lsst_16']:.0f}-{v['N_lsst_84']:.0f})"
                                        for k, v in out_v.items()))
    _save("lsst_yield", out)


# ---------------------------------------------------------------- N-body H1
def _kappa_from_R(R):
    """von Mises concentration with mean resultant length R (Banerjee et al. 2005 approximation)."""
    R = np.clip(R, 1e-6, 0.95)
    return R * (2 - R**2) / (1 - R**2)


def _fisher_kappa(mean_cos):
    """Fisher concentration with <cos theta> = coth k - 1/k, solved numerically."""
    from scipy.optimize import brentq
    m = float(np.clip(mean_cos, 1e-4, 0.995))
    return brentq(lambda k: 1 / np.tanh(k) - 1 / k - m, 1e-4, 1e4)


def stage_nbody_k(n_grid=72):
    """Bayes factor of each simulated planet, with the orientation density taken from the
    N-body runs (per-a-bin von Mises in varpi - varpi_9 about the simulated cluster centre,
    Fisher pole scatter) and the planet's (varpi_9, Omega_9) marginalised on a uniform grid."""
    from planetix.inference.model import _log_vm_rel

    df = load_etnos("bb16")
    S = selection_cube("bb16")
    nv, no = S.shape[1:]
    gv = np.radians((np.arange(nv) + 0.5) * 360.0 / nv)
    go = np.radians((np.arange(no) + 0.5) * 360.0 / no)
    out = {}
    for fname in ("nbody_population_neptune", "nbody_population"):
        path = OUT / f"{fname}.json"
        if not path.exists():
            continue
        for name, r in json.loads(path.read_text()).items():
            prof = r["profile"]
            edges = np.array([p["a_lo"] for p in prof] + [prof[-1]["a_hi"]])
            k = np.clip(np.searchsorted(edges, df["a"].to_numpy()) - 1, 0, len(prof) - 1)
            R = np.array([prof[j]["R_varpi"] for j in k])
            mu = np.radians([prof[j]["mean_dvarpi"] for j in k])
            kv = _kappa_from_R(R)
            kp = np.array([_fisher_kappa(prof[j]["mean_cos_pole"]) for j in k])
            i9 = r["planet"]["i9"]
            ko = kp * np.sin(np.radians(df["i"].to_numpy())) * np.sin(np.radians(i9))
            v9 = np.radians((np.arange(n_grid) + 0.5) * 360.0 / n_grid)
            o9 = v9.copy()
            x_v, x_o = np.radians(df["varpi"].to_numpy()), np.radians(df["Omega"].to_numpy())
            # log g at the data, (n_grid varpi9, n_obj) and (n_grid Omega9, n_obj)
            lv = _log_vm_rel(x_v[None, :] - v9[:, None] - mu[None, :], kv[None, :])
            lo = _log_vm_rel(x_o[None, :] - o9[:, None], ko[None, :])
            # normalisation sum_x g S for every (varpi9, Omega9): separable in the two angles
            hv = np.exp(_log_vm_rel(gv[None, None, :] - v9[:, None, None] - mu[None, :, None], kv[None, :, None]))
            ho = np.exp(_log_vm_rel(go[None, None, :] - o9[:, None, None], ko[None, :, None]))
            Z = np.einsum("ajv,jvo,bjo->abj", hv, S, ho)            # (varpi9, Omega9, obj)
            logr = lv[:, None, :] + lo[None, :, :] - np.log(Z)
            tot = logr.sum(axis=2)
            ln_k = float(np.log(np.mean(np.exp(tot - tot.max()))) + tot.max())
            best = np.unravel_index(np.argmax(tot), tot.shape)
            out[f"{fname}:{name}"] = {"ln_bf": ln_k, "varpi9_best": float(np.degrees(v9[best[0]])),
                                      "Omega9_best": float(np.degrees(o9[best[1]]))}
            print(f"  {fname} {name}: lnK = {ln_k:.2f}  (best varpi9 {np.degrees(v9[best[0]]):.0f}, "
                  f"Omega9 {np.degrees(o9[best[1]]):.0f})")
    _save("nbody_bayes_factor", out)


STAGES = {"history": stage_history, "predictive": stage_predictive, "selmarg": stage_selmarg,
          "lsst": stage_lsst, "lsstsim": stage_lsstsim, "yield": stage_yield, "nbody_k": stage_nbody_k}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    for s in args.stages:
        print(f"[{s}]")
        fn = STAGES[s]
        fn(n_proc=args.procs) if "n_proc" in fn.__code__.co_varnames else fn()


if __name__ == "__main__":
    main()
