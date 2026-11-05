"""N-body mean orbit poles of ETNO-like test particles under a distant planet.

For each planet configuration and each (a, q) bin, integrates a small group of
test particles (Sun + giants-as-J2 + planet, REBOUND IAS15, real masses, no
boost) and records each surviving particle's time-averaged orbit pole. The
secular model's prediction is written alongside so the two can be compared.

    python scripts/nbody_forced_poles.py --out data/processed/nbody_forced_poles.csv

Takes about 45 minutes on four cores with the defaults.
"""

import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from planetix.dynamics.frames import angle_between, pole_to_angles
from planetix.dynamics.nbody import integrate
from planetix.dynamics.secular import Perturber, forced_pole, giant_planets

PLANETS = {
    "5Me_500AU_i20": Perturber.earths(5.0, 500.0, 20.0, 0.0, 0.25),
    "10Me_700AU_i20": Perturber.earths(10.0, 700.0, 20.0, 0.0, 0.25),
}
A_BINS = [200.0, 250.0, 300.0, 400.0, 550.0, 750.0]
Q_PERI = 45.0
N_PER = 6


def run(task):
    key, a, t_end, seed = task
    planet = PLANETS[key]
    e = 1.0 - Q_PERI / a
    pert = giant_planets() + [planet]
    f = forced_pole(a, pert, e, "nonlinear", "quadrupole")
    i0, O0 = pole_to_angles(f)
    nodes = O0 + np.linspace(0, 360, N_PER, endpoint=False)
    r = integrate([a] * N_PER, e, [i0 + 5.0] * N_PER, nodes, planet, t_end, n_out=500, seed=seed)
    ok = r.survived([a] * N_PER, tol=0.3)
    mp = r.mean_poles(t_min=0.2 * t_end)
    rows = []
    for k in range(N_PER):
        i_m, O_m = pole_to_angles(mp[k])
        rows.append(dict(planet=key, a=a, e=e, particle=k, survived=bool(ok[k]), i_mean=i_m, Omega_mean=O_m,
                         i_secular=float(i0), Omega_secular=float(O0), offset_deg=float(angle_between(mp[k], f)),
                         a_final=float(r.a[-1, k]), t_end=t_end))
    print(key, a, "survivors", ok.sum(), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/nbody_forced_poles.csv")
    ap.add_argument("--t-end", type=float, default=1e9)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    tasks = [(k, a, args.t_end, s) for s, (k, a) in enumerate(itertools.product(PLANETS, A_BINS))]
    with Pool(args.procs) as pool:
        rows = sum(pool.map(run, tasks, chunksize=1), [])
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
