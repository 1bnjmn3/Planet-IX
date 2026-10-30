"""Fit a distant planet to the ETNO orbit poles and write what the data constrain.

    python scripts/invert_planet.py [--mocks 100] [--out data/processed/inversion]

Writes, under --out:
  mean_poles.csv        mean orbit pole per a-bin with bootstrap and null p-values
  profile_<cut>.csv     profile likelihood over (mass, a_p) for free and fixed i_p
  upper_limits.csv      95% one-sided mass limits per (cut, a_p, i_p)
  calibration.json      test statistics on the data and on no-planet mocks
  script25.json         the original script 25 fit, on data and isotropic mocks
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from planetix.data import load_catalog, load_etnos
from planetix.dynamics.frames import invariable_to_ecliptic, pole_to_angles
from planetix.dynamics.inversion import (
    Sample, circular_fit_demo, fit_common_plane, fit_null, mock_null, profile_grid, sample_vmf,
    lower_limits, test_statistic, upper_limits,
)

MASSES = np.geomspace(0.05, 50.0, 19)
A_PLANETS = [400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0]
I_FIXED = [None, 10.0, 20.0, 30.0]
A_BINS = [(150, 250), (250, 400), (400, 700), (700, np.inf), (150, np.inf), (250, np.inf)]


def mean_pole(p):
    m = p.mean(axis=0)
    return m / np.linalg.norm(m)


def mean_pole_table(s, rng, n_boot=2000, n_null=4000):
    rows = []
    for lo, hi in A_BINS:
        k = (s.a > lo) & (s.a <= hi)
        P = s.poles[k]
        i, O = pole_to_angles(mean_pole(P))
        ie, Oe = invariable_to_ecliptic(i, O)
        boot = [pole_to_angles(mean_pole(P[rng.integers(0, len(P), len(P))]))[0] for _ in range(n_boot)]
        kap = fit_null(Sample(P, s.a[k], s.e[k]))["kappa"]
        null = [pole_to_angles(mean_pole(sample_vmf([0, 0, 1], kap, len(P), rng)))[0] for _ in range(n_null)]
        rows.append(dict(a_min=lo, a_max=hi, n=int(k.sum()), i_inv=i, Omega_inv=O, i_ecl=ie, Omega_ecl=Oe,
                         i_lo68=np.percentile(boot, 16), i_hi68=np.percentile(boot, 84),
                         p_null=float(np.mean(np.asarray(null) >= i))))
    return pd.DataFrame(rows)


def common_plane_ts(s):
    return 2.0 * (fit_common_plane(s)["loglike"] - fit_null(s)["loglike"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mocks", type=int, default=100)
    ap.add_argument("--out", default="data/processed/inversion")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    samples = {cut: Sample.from_frame(load_etnos(cut)) for cut in ("ts14", "bb16")}
    pd.concat([mean_pole_table(s, rng).assign(cut=c) for c, s in samples.items()]).to_csv(out / "mean_poles.csv", index=False)

    limits, calib = [], {}
    for cut, s in samples.items():
        rows = []
        for i_fix in I_FIXED:
            g = profile_grid(s, MASSES, A_PLANETS, i_fixed=i_fix)
            ul, ll = upper_limits(g), lower_limits(g)
            for ia, a_p in enumerate(A_PLANETS):
                limits.append(dict(cut=cut, a_p=a_p, i_p=i_fix if i_fix is not None else "free", n=int(g["n"][ia]),
                                   mass_upper_95=ul[ia], mass_lower_95=ll[ia],
                                   best_2dlnL=2 * (np.nanmax(g["loglike"][ia]) - g["null_loglike"][ia])
                                   if g["n"][ia] >= 5 else np.nan))
                for im, m in enumerate(MASSES):
                    rows.append(dict(i_fixed=i_fix, a_p=a_p, mass=m, n=int(g["n"][ia]),
                                     dlnL=g["loglike"][ia, im] - g["null_loglike"][ia],
                                     i_p=g["i_p"][ia, im], Omega_p=g["Omega_p"][ia, im], kappa=g["kappa"][ia, im]))
        pd.DataFrame(rows).to_csv(out / f"profile_{cut}.csv", index=False)

        # data and mocks use the same (coarser) mass grid so their test statistics are comparable
        ts_planet = test_statistic(profile_grid(s, MASSES[::3], A_PLANETS))
        kappa = fit_null(s)["kappa"]
        mocks_planet, mocks_plane = [], []
        for _ in range(args.mocks):
            m = mock_null(s, kappa, rng)
            mocks_planet.append(test_statistic(profile_grid(m, MASSES[::3], A_PLANETS)))
            mocks_plane.append(common_plane_ts(m))
        ts_plane = common_plane_ts(s)
        cp = fit_common_plane(s)
        calib[cut] = dict(
            n=len(s), kappa_null=kappa,
            common_plane=dict(i_inv=cp["i"], Omega_inv=cp["Omega"], ts=ts_plane,
                              p_mock=float(np.mean(np.asarray(mocks_plane) >= ts_plane))),
            planet=dict(ts=ts_planet, p_mock=float(np.mean(np.asarray(mocks_planet) >= ts_planet)),
                        mock_ts_median=float(np.median(mocks_planet)), mock_ts_95=float(np.percentile(mocks_planet, 95))),
            n_mocks=args.mocks,
        )
    pd.DataFrame(limits).to_csv(out / "upper_limits.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))

    cat = load_catalog()
    demo = circular_fit_demo(cat[(cat["a"] > 150) & (cat["q"] > 30)], n_mock=300)
    mi, mm = demo["mock_inclination"], demo["mock_mass"]
    (out / "script25.json").write_text(json.dumps(dict(
        real=dict(i_planet=demo["real"][0], mass_earth=demo["real"][1], n=demo["real"][2]),
        mocks=dict(n_mock=300, frac_fitted=float(np.mean(~np.isnan(mi))),
                   i_planet_median=float(np.nanmedian(mi)), i_planet_5_95=list(np.nanpercentile(mi, [5, 95])),
                   mass_median=float(np.nanmedian(mm)), mass_5_95=list(np.nanpercentile(mm, [5, 95])),
                   frac_mass_at_bound=float(np.nanmean((mm < 0.11) | (mm > 19.9)))),
    ), indent=2))
    print(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
