"""Gigayear N-body forward model of the ETNO population a distant planet sculpts.

Sun (carrying the giants' mass) + one distant planet + massless test particles,
integrated with REBOUND's WHFast. The giant planets enter as the orbit-averaged
quadrupole they exert on distant bodies, J2 R^2 = (1/2) sum m_j a_j^2
(``constants.giants_j2r2``). Instead of a direct J2 force, which is sharply
peaked at perihelion and would need a short timestep, its secular effect is
applied exactly as a rotation of each particle's state vector every
``t_secular`` years:

    dOmega/dt = -(3/2) n J2R^2 / p^2 cos i,
    domega/dt =  (3/4) n J2R^2 / p^2 (5 cos^2 i - 1),     p = a (1 - e^2).

Both are rigid rotations of (r, v): about the reference z axis for Omega and
about the orbit normal for omega, so the Keplerian elements other than the two
angles are untouched. With the Sun-planet system as the only short-period
dynamics, a 50 yr step (>= 35 steps per orbit at a >= 150 AU) runs a gigayear
of 1000 particles in about an hour on one core.

Neptune's scattering is represented by removing particles once their
perihelion falls below ``q_sink`` (30 AU by default), as in Batygin & Brown
(2016). Particles beyond ``a_max`` or on unbound orbits are removed too.

This gives the distribution of (varpi - varpi_9, orbit pole) as a function of
a, which ``planetix.inference.model`` otherwise assumes: a von Mises cluster
anti-aligned with the planet plus an isotropic background, switched on above
a_crit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetix.dynamics.constants import GIANTS, GM_SUN, M_EARTH, giants_j2r2


@dataclass
class PlanetSpec:
    m9: float          # Earth masses
    a9: float          # AU
    e9: float
    i9: float          # deg, to the reference (giant-planet) plane
    varpi9: float = 0.0
    Omega9: float = 0.0
    name: str = ""


@dataclass
class PopulationRun:
    planet: PlanetSpec
    times: np.ndarray           # yr, (T,)
    elements: np.ndarray        # (T, N, 6): a, e, i, Omega, omega, M (deg), NaN once removed
    initial: np.ndarray         # (N, 6) initial elements

    def snapshot(self, t_min: float):
        """Stacked elements of all live particles at times >= t_min, shape (K, 6)."""
        sel = self.times >= t_min
        el = self.elements[sel].reshape(-1, 6)
        return el[np.isfinite(el[:, 0])]


def _rotate(vec, axis, angle):
    """Rodrigues rotation of vectors (N, 3) about unit axes (N, 3) by angles (N,)."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return vec * c + np.cross(axis, vec) * s + axis * np.sum(axis * vec, axis=1, keepdims=True) * (1 - c)


def _elements(r, v, mu):
    """a, e, i, Omega, omega, M (deg) and orbit normal from heliocentric state vectors."""
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    rn = np.linalg.norm(r, axis=1)
    v2 = np.sum(v * v, axis=1)
    a = 1.0 / (2.0 / rn - v2 / mu)
    ev = np.cross(v, h) / mu - r / rn[:, None]
    e = np.linalg.norm(ev, axis=1)
    inc = np.degrees(np.arccos(np.clip(h[:, 2] / hn, -1, 1)))
    Om = np.degrees(np.arctan2(h[:, 0], -h[:, 1])) % 360.0
    node = np.stack([np.cos(np.radians(Om)), np.sin(np.radians(Om)), np.zeros_like(Om)], axis=1)
    hhat = h / hn[:, None]
    # argument of perihelion: angle from the node to the eccentricity vector, in the orbit plane
    om = np.degrees(np.arctan2(np.sum(np.cross(node, ev) * hhat, axis=1), np.sum(node * ev, axis=1))) % 360.0
    f = np.arctan2(np.sum(np.cross(ev, r) * hhat, axis=1), np.sum(ev * r, axis=1))
    with np.errstate(invalid="ignore"):
        E = 2 * np.arctan(np.sqrt(np.clip((1 - e) / (1 + e), 0, None)) * np.tan(f / 2))
    M = np.degrees(E - e * np.sin(E)) % 360.0
    return np.stack([a, e, inc, Om, om, M], axis=1), hhat


def initial_conditions(n, rng, a_range=(150.0, 1500.0), q_range=(30.0, 50.0), i_sigma=15.0):
    """Log-uniform a, uniform q, i from sin(i) x half-normal(i_sigma), random angles."""
    a = np.exp(rng.uniform(*np.log(a_range), n))
    q = rng.uniform(*q_range, n)
    inc = np.empty(n)
    k = 0
    while k < n:  # rejection sample sin(i) exp(-i^2 / 2 sigma^2)
        x = np.abs(rng.normal(0, i_sigma, 4 * n))
        x = x[(x < 90) & (rng.random(x.size) < np.sin(np.radians(x)))]
        take = min(n - k, x.size)
        inc[k:k + take] = x[:take]
        k += take
    return np.stack([a, 1 - q / a, inc, rng.uniform(0, 360, n), rng.uniform(0, 360, n), rng.uniform(0, 360, n)], axis=1)


def run_population(planet: PlanetSpec | None, n_particles=1000, t_end=1.0e9, n_out=1000, dt=50.0,
                   t_secular=1.0e4, q_sink=30.0, a_max=1.0e4, seed=0, progress=None) -> PopulationRun:
    import rebound

    rng = np.random.default_rng(seed)
    init = initial_conditions(n_particles, rng)
    m_sun = 1.0 + sum(m for m, _ in GIANTS.values())
    mu = GM_SUN * m_sun
    j2r2 = giants_j2r2()

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=m_sun)
    n_massive = 1
    if planet is not None:
        sim.add(m=planet.m9 * M_EARTH, a=planet.a9, e=planet.e9, inc=np.radians(planet.i9),
                Omega=np.radians(planet.Omega9), omega=np.radians((planet.varpi9 - planet.Omega9) % 360.0), M=0.0)
        n_massive = 2
    for k, (a, e, inc, Om, om, M) in enumerate(init):
        sim.add(primary=sim.particles[0], a=a, e=e, inc=np.radians(inc), Omega=np.radians(Om),
                omega=np.radians(om), M=np.radians(M))
    sim.N_active = n_massive
    sim.testparticle_type = 0
    sim.integrator = "whfast"
    sim.dt = dt
    sim.move_to_com()

    times = np.linspace(0.0, t_end, n_out + 1)[1:]
    out = np.full((n_out, n_particles, 6), np.nan)
    t = 0.0
    alive = np.arange(n_particles)  # particle k is sim.particles[n_massive + position]
    for t_idx, t_out in enumerate(times):
        while t < t_out - 1e-6:
            t_next = min(t + t_secular, t_out)
            sim.integrate(t_next, exact_finish_time=0)
            step = sim.t - t
            t = sim.t
            # secular J2 of the giants, as rotations of the heliocentric state vectors
            buf = np.zeros((sim.N, 6))
            sim.serialize_particle_data(xyzvxvyvz=buf)
            sun = buf[0].copy()
            r = buf[n_massive:, :3] - sun[:3]
            v = buf[n_massive:, 3:] - sun[3:]
            el, hhat = _elements(r, v, mu)
            a, e, inc = el[:, 0], el[:, 1], np.radians(el[:, 2])
            p = np.abs(a) * np.abs(1 - e**2)
            n = np.sqrt(mu / np.abs(a) ** 3)
            fac = n * j2r2 / p**2 * step
            dOm = -1.5 * fac * np.cos(inc)
            dom = 0.75 * fac * (5 * np.cos(inc) ** 2 - 1)
            r, v = _rotate(r, hhat, dom), _rotate(v, hhat, dom)
            z = np.tile([0.0, 0.0, 1.0], (len(r), 1))
            r, v = _rotate(r, z, dOm), _rotate(v, z, dOm)
            buf[n_massive:, :3] = r + sun[:3]
            buf[n_massive:, 3:] = v + sun[3:]
            sim.set_serialized_particle_data(xyzvxvyvz=buf)
        # record and remove
        buf = np.zeros((sim.N, 6))
        sim.serialize_particle_data(xyzvxvyvz=buf)
        r = buf[n_massive:, :3] - buf[0, :3]
        v = buf[n_massive:, 3:] - buf[0, 3:]
        el, _ = _elements(r, v, mu)
        q = el[:, 0] * (1 - el[:, 1])
        dead = (el[:, 1] >= 1) | (el[:, 0] <= 0) | (el[:, 0] > a_max) | (q < q_sink)
        out[t_idx, alive[~dead]] = el[~dead]
        for pos in np.flatnonzero(dead)[::-1]:
            sim.remove(int(n_massive + pos))
        alive = alive[~dead]
        if progress is not None:
            progress(t_out, len(alive))
        if not len(alive):
            break
    return PopulationRun(planet, times, out, init)


def confinement_profile(run: PopulationRun, t_min: float, a_edges, q_min=30.0):
    """Mean-resultant statistics of varpi - varpi_9 and of the pole offset from the
    planet's pole, per semi-major-axis bin, over all snapshots after t_min."""
    el = run.snapshot(t_min)
    el = el[el[:, 0] * (1 - el[:, 1]) > q_min]
    p = run.planet
    dvarpi = np.radians((el[:, 3] + el[:, 4] - p.varpi9) % 360.0)
    pole = _pole(el[:, 2], el[:, 3])
    pole9 = _pole(np.array([p.i9]), np.array([p.Omega9]))[0]
    rows = []
    for lo, hi in zip(a_edges[:-1], a_edges[1:]):
        m = (el[:, 0] >= lo) & (el[:, 0] < hi)
        if m.sum() < 20:
            continue
        R = np.hypot(np.cos(dvarpi[m]).mean(), np.sin(dvarpi[m]).mean())
        mean_dir = np.degrees(np.arctan2(np.sin(dvarpi[m]).mean(), np.cos(dvarpi[m]).mean())) % 360.0
        cosang = pole[m] @ pole9
        rows.append({"a_lo": lo, "a_hi": hi, "n": int(m.sum()), "R_varpi": R, "mean_dvarpi": mean_dir,
                     "mean_pole_offset_deg": float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean()),
                     "mean_cos_pole": float(cosang.mean())})
    return rows


def _pole(i_deg, Omega_deg):
    i, O = np.radians(i_deg), np.radians(Omega_deg)
    return np.stack([np.sin(i) * np.sin(O), -np.sin(i) * np.cos(O), np.cos(i)], axis=-1)
