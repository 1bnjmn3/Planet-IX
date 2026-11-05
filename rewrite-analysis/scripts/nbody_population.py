"""Gigayear N-body forward models of the ETNO population sculpted by a distant planet.

Runs ``planetix.dynamics.population.run_population`` for each planet below (one
process each) and writes the element histories to paper/cache/ and a
confinement profile per semi-major-axis bin to paper/results/nbody_population.json.

    python scripts/nbody_population.py                  # ~1.7 h per planet on one core
    python scripts/nbody_population.py --n 200 --t-end 5e7   # quick check
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from planetix.dynamics.population import PlanetSpec, confinement_profile, run_population

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "results"
CACHE = ROOT / "paper" / "cache"

PLANETS = {
    # Brown & Batygin (2021) maximum-likelihood planet
    "BB21": PlanetSpec(6.2, 380.0, 0.20, 16.0, 247.0, 96.0, "BB21"),
    # reference planet that normalises a_crit in planetix.inference.model
    "REF": PlanetSpec(5.0, 500.0, 0.25, 20.0, 247.0, 96.0, "REF"),
    # Batygin & Brown (2016) original proposal
    "BB16": PlanetSpec(10.0, 700.0, 0.60, 30.0, 247.0, 96.0, "BB16"),
}
A_EDGES = [150, 200, 250, 300, 400, 550, 750, 1000, 1500]


def _run(args):
    name, n, t_end, seed, neptune = args
    tag = "_neptune" if neptune else ""
    kw = dict(neptune=True, dt=10.0, q_sink=20.0) if neptune else {}
    run = run_population(PLANETS[name], n_particles=n, t_end=t_end, n_out=500, seed=seed,
                         checkpoint=CACHE / f"nbody_{name}{tag}_partial.npz", **kw)
    np.savez_compressed(CACHE / f"nbody_{name}{tag}.npz", times=run.times, elements=run.elements.astype(np.float32),
                        initial=run.initial, planet_elements=run.planet_elements)
    prof = confinement_profile(run, t_min=0.5 * t_end, a_edges=A_EDGES)
    alive = int(np.isfinite(run.elements[-1, :, 0]).sum())
    return name, {"planet": vars(PLANETS[name]), "n_particles": n, "t_end": t_end, "n_alive_end": alive,
                  "neptune_explicit": neptune, "profile": prof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--t-end", type=float, default=1.0e9)
    ap.add_argument("--planets", nargs="*", default=list(PLANETS))
    ap.add_argument("--neptune", action="store_true",
                    help="integrate Neptune explicitly (MERCURIUS, dt = 10 yr) instead of a q < 30 au sink")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(len(args.planets)) as pool:
        res = dict(pool.map(_run, [(p, args.n, args.t_end, k, args.neptune) for k, p in enumerate(args.planets)]))
    name = "nbody_population_neptune" if args.neptune else "nbody_population"
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    for name, r in res.items():
        print(name, r["n_alive_end"])
        for row in r["profile"]:
            print(f"  a {row['a_lo']}-{row['a_hi']}: n={row['n']} R={row['R_varpi']:.2f} "
                  f"<dvarpi>={row['mean_dvarpi']:.0f} pole offset {row['mean_pole_offset_deg']:.0f}")


if __name__ == "__main__":
    main()
