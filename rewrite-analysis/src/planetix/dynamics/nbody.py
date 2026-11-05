"""N-body cross-check of the secular model with REBOUND.

The giant planets are replaced by the solar J2 they induce on distant orbits
(``giants_j2r2``; Sun mass increased by their total mass), as in Batygin &
Brown (2016). This keeps the short-period giants out of the integration so a
gigayear-scale secular cycle runs in minutes, and is exact to leading order for
particles with q >> 30 AU. Requires ``rebound`` and ``reboundx`` (``pip install
planetix[nbody]``).

The time-averaged orbit pole of a test particle is its forced pole once the
free precession has completed several cycles, so ``measure_forced_poles``
returns the quantity ``secular.forced_pole`` predicts.

``boost`` multiplies the planet mass and the giants' J2 by the same factor.
Linear secular rates scale with mass, so the forced pole is unchanged while the
secular period shrinks by ``boost``; use it only where the boosted planet stays
far from the particles (scattering is not linear in mass).

Replaces original script 19, which integrated 500 kyr (well under 1% of one
secular period) from random orbits and called particles within 20 deg of the
target "captured".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetix.dynamics.constants import GIANTS, giants_j2r2
from planetix.dynamics.secular import Perturber, giant_planets, nodal_precession_rate


@dataclass
class NBodyResult:
    times: np.ndarray        # yr, shape (T,)
    poles: np.ndarray        # unit orbit normals, shape (T, N, 3), invariable frame
    a: np.ndarray            # osculating a, shape (T, N)
    e: np.ndarray            # osculating e, shape (T, N)

    def mean_poles(self, t_min=0.0):
        """Time-averaged pole of each particle (renormalised), shape (N, 3)."""
        sel = self.times >= t_min
        m = self.poles[sel].mean(axis=0)
        return m / np.linalg.norm(m, axis=-1, keepdims=True)

    def survived(self, a0, tol=0.2):
        """Particles whose a stayed within a fractional ``tol`` of ``a0`` (no scattering)."""
        return np.all(np.abs(self.a / np.asarray(a0) - 1.0) < tol, axis=0)


def _sim(planet: Perturber | None, boost: float, r_eq: float):
    import rebound
    import reboundx

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    m_giants = sum(m for m, _ in GIANTS.values())
    sim.add(m=1.0 + m_giants, r=r_eq)
    if planet is not None:
        sim.add(m=planet.mass * boost, a=planet.a, e=planet.e, inc=np.radians(planet.i),
                Omega=np.radians(planet.Omega), omega=0.0, M=0.0)
    rebx = reboundx.Extras(sim)
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    sim.particles[0].params["J2"] = boost * giants_j2r2() / r_eq**2
    sim.particles[0].params["R_eq"] = r_eq
    return sim, rebx


def integrate(a, e, i, Omega, planet: Perturber | None, t_end, n_out=400, boost=1.0, seed=0, r_eq=1.0):
    """Integrate test particles with initial (a, e, i, Omega) arrays (deg, invariable frame).

    Arguments of perihelion and mean anomalies are random. Returns ``NBodyResult``.
    """
    a, e, i, Omega = (np.atleast_1d(np.asarray(x, float)) for x in np.broadcast_arrays(a, e, i, Omega))
    rng = np.random.default_rng(seed)
    sim, rebx = _sim(planet, boost, r_eq)
    sun = sim.particles[0]
    n_massive = sim.N
    for k in range(len(a)):
        sim.add(primary=sun, a=a[k], e=e[k], inc=np.radians(i[k]), Omega=np.radians(Omega[k]),
                omega=rng.uniform(0, 2 * np.pi), M=rng.uniform(0, 2 * np.pi))
    sim.N_active = n_massive
    sim.testparticle_type = 0
    sim.integrator = "ias15"
    sim.move_to_com()

    times = np.linspace(0.0, t_end, n_out)
    poles = np.empty((n_out, len(a), 3))
    aa = np.empty((n_out, len(a)))
    ee = np.empty((n_out, len(a)))
    for t_idx, t in enumerate(times):
        sim.integrate(t, exact_finish_time=0)
        sun = sim.particles[0]
        for k in range(len(a)):
            p = sim.particles[n_massive + k]
            r = np.array(p.xyz) - np.array(sun.xyz)
            v = np.array(p.vxyz) - np.array(sun.vxyz)
            h = np.cross(r, v)
            poles[t_idx, k] = h / np.linalg.norm(h)
            o = p.orbit(primary=sun)
            aa[t_idx, k], ee[t_idx, k] = o.a, o.e
    return NBodyResult(times, poles, aa, ee)


def secular_period(a, planet: Perturber | None, e=0.0, ecc=None, boost=1.0):
    """Free nodal period (yr) predicted by the secular model, for choosing ``t_end``."""
    pert = giant_planets() + ([planet] if planet is not None else [])
    return 2 * np.pi / abs(nodal_precession_rate(a, pert, e, ecc)) / boost
