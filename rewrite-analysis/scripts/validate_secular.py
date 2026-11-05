"""Compare the secular forced inclination with REBOUND in and out of its domain.

    python scripts/validate_secular.py --out data/processed/secular_vs_nbody.csv

Planet: 5 Me, a = 500 AU, i = 20 deg (invariable frame). Masses and the giants'
J2 are boosted 10x so each secular cycle takes tens of Myr (the forced pole is
independent of the boost in the linear regime). Four particles per row, started
3 deg from the forced pole at four node phases, integrated for four free periods.
Takes about 6 minutes.
"""

import argparse

import numpy as np
import pandas as pd

from planetix.dynamics.frames import pole_to_angles
from planetix.dynamics.nbody import integrate, secular_period
from planetix.dynamics.secular import Perturber, forced_pole, giant_planets

CASES = [(150, 0.2), (200, 0.3), (250, 0.3), (300, 0.2), (200, 0.7), (250, 0.7), (300, 0.85)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/secular_vs_nbody.csv")
    ap.add_argument("--boost", type=float, default=10.0)
    args = ap.parse_args()
    planet = Perturber.earths(5.0, 500.0, 20.0, 100.0)
    pert = giant_planets() + [planet]
    rows = []
    for a, e in CASES:
        T = secular_period(a, planet, e, "quadrupole", boost=args.boost)
        lin = forced_pole(a, pert, e, "linear")
        nl = forced_pole(a, pert, e, "nonlinear")
        qe = forced_pole(a, pert, e, "nonlinear", "quadrupole")
        i0, O0 = pole_to_angles(nl)
        r = integrate([a] * 4, e, [i0 + 3] * 4, O0 + np.array([0, 90, 180, 270]), planet, 4 * T,
                      n_out=400, boost=args.boost)
        ok = r.survived([a] * 4)
        m = r.mean_poles()[ok].mean(axis=0)
        Q = a * (1 + e)
        rows.append(dict(a=a, e=e, Q=Q, crosses=bool(Q >= 0.8 * planet.a), period_myr=T / 1e6, survivors=int(ok.sum()),
                         i_nbody=pole_to_angles(m)[0], i_ll_linear=pole_to_angles(lin)[0],
                         i_ll_nonlinear=pole_to_angles(nl)[0], i_quadrupole_ecc=pole_to_angles(qe)[0]))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
