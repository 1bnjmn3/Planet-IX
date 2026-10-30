"""What the ETNO orbit poles constrain about a distant planet.

Replaces original script 25 ("WarpInverter"), which kept only objects with
14 < i < 22 deg and then fitted their inclinations: a sample selected to sit at
i ~ 18 deg returns i ~ 18 deg whatever the sky contains (``circular_fit_demo``
shows this on isotropic mock data). Here every object passing the sample cuts
is used, and nothing is selected on the angles being fitted.

Model
-----
Each ETNO pole ``l_k`` (unit vector, invariable frame) scatters about its forced
pole ``f_k = f(a_k, e_k; m, a_p, i_p, Omega_p)`` from ``secular`` with a
von Mises-Fisher density of concentration ``kappa`` (free inclinations with
random free nodes). The null hypothesis is no planet: ``f_k`` is the invariable
pole for every object. A model-free alternative, ``common_plane``, lets all
poles share one free mean pole.

Because the planet's (i_p, Omega_p) are undefined when m -> 0, Wilks' theorem
does not apply to the planet-vs-null likelihood ratio; ``null_distribution``
calibrates it with mock samples drawn from the fitted null.

Not modelled here: survey selection (pointings near the ecliptic favour some
(i, Omega)), the giants' plane being torqued by the planet, and orbits that
cross the planet's (q < a_p < Q), where secular theory does not apply.
``valid_mask`` drops those per a_p and ``crossing_fraction`` reports how many.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from planetix.dynamics.frames import ecliptic_to_invariable, pole, pole_to_angles
from planetix.dynamics.secular import Perturber, coupling_rate, giant_planets, two_plane_tilt


# ---------------------------------------------------------------- likelihood

def vmf_loglike(poles, centres, kappa):
    """Summed von Mises-Fisher log density of unit vectors ``poles`` about ``centres``."""
    c = np.sum(poles * centres, axis=-1)
    # log(kappa / (4 pi sinh kappa)), written to stay finite for large kappa
    log_norm = np.log(kappa) - np.log(2 * np.pi) - kappa - np.log1p(-np.exp(-2 * kappa))
    return float(np.sum(log_norm + kappa * c))


def _best_kappa(poles, centres):
    r = float(np.mean(np.sum(poles * centres, axis=-1)))
    res = minimize_scalar(lambda lk: -vmf_loglike(poles, centres, np.exp(lk)), bounds=(-5, 9), method="bounded")
    return float(np.exp(res.x)), -float(res.fun), r


@dataclass
class Sample:
    """ETNO poles in the invariable frame plus the (a, e) that set their forcing."""

    poles: np.ndarray
    a: np.ndarray
    e: np.ndarray
    names: tuple = ()

    @classmethod
    def from_frame(cls, df):
        i, O = ecliptic_to_invariable(df["i"].to_numpy(), df["Omega"].to_numpy())
        names = tuple(df["designation"]) if "designation" in df else ()
        return cls(pole(i, O), df["a"].to_numpy(float), df["e"].to_numpy(float), names)

    def __len__(self):
        return len(self.a)


class PlanetModel:
    """Forced poles for giants + one planet on a circular-or-eccentric orbit.

    Coupling weights depend only on (m, a_p, e_p), so they are cached per grid
    point; (i_p, Omega_p) then enter through the closed-form two-plane tilt.
    """

    def __init__(self, sample: Sample, ecc="quadrupole", e_planet=0.0):
        self.s, self.ecc, self.e_planet = sample, ecc, e_planet
        giants = giant_planets()
        self.w_ref = np.array([sum(coupling_rate(a, g, e, ecc) for g in giants) for a, e in zip(sample.a, sample.e)])

    def planet_weights(self, mass_earth, a_planet):
        p = Perturber.earths(mass_earth, a_planet, e=self.e_planet)
        return np.array([coupling_rate(a, p, e, self.ecc) for a, e in zip(self.s.a, self.s.e)])

    def forced(self, w_planet, i_p, Omega_p):
        phi = np.radians(two_plane_tilt(self.w_ref, w_planet, i_p))
        O = np.radians(Omega_p)
        # rotate the invariable pole towards the planet pole by phi, about the planet's node line
        return np.stack([np.sin(phi) * np.sin(O), -np.sin(phi) * np.cos(O), np.cos(phi)], axis=-1)


# ---------------------------------------------------------------- fits

def fit_null(sample: Sample):
    """No planet: poles scatter about the invariable pole. Returns dict(loglike, kappa)."""
    z = np.broadcast_to([0.0, 0.0, 1.0], sample.poles.shape)
    kappa, ll, _ = _best_kappa(sample.poles, z)
    return {"loglike": ll, "kappa": kappa}


def fit_common_plane(sample: Sample):
    """All poles about one free mean pole (two extra parameters)."""
    m = sample.poles.mean(axis=0)
    m /= np.linalg.norm(m)
    kappa, ll, _ = _best_kappa(sample.poles, np.broadcast_to(m, sample.poles.shape))
    i, O = pole_to_angles(m)
    return {"loglike": ll, "kappa": kappa, "i": float(i), "Omega": float(O), "pole": m}


def _fit_orientation(model: PlanetModel, w_planet, starts, i_fixed=None):
    poles = model.s.poles

    def nll(x):
        i_p, O_p = (x[0], x[1]) if i_fixed is None else (i_fixed, x[0])
        return -_best_kappa(poles, model.forced(w_planet, i_p, O_p))[1]

    best = None
    for x0 in starts:
        x0 = x0 if i_fixed is None else x0[1:]
        r = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    if i_fixed is None:
        i_p, O_p = best.x[0], best.x[1] % 360.0
    else:
        i_p, O_p = i_fixed, best.x[0] % 360.0
    if i_p < 0:
        i_p, O_p = -i_p, (O_p + 180.0) % 360.0
    f = model.forced(w_planet, i_p, O_p)
    k, ll, _ = _best_kappa(poles, f)
    return {"loglike": ll, "kappa": k, "i_p": float(i_p), "Omega_p": float(O_p)}


def _orientation_starts(sample):
    cp = fit_common_plane(sample)
    return [(max(cp["i"], 1.0), cp["Omega"]), (20.0, cp["Omega"]), (30.0, (cp["Omega"] + 180) % 360),
            (10.0, 90.0), (10.0, 270.0)]


def valid_mask(sample: Sample, a_planet, inner=0.8, outer=1.25):
    """Objects that stay clear of the planet's orbit: Q < inner * a_p or q > outer * a_p.

    The coupling rates assume non-crossing orbits; ``nbody`` shows the secular
    forced pole is wrong by a factor ~2 for orbits that reach the planet. The
    mask depends only on (a, e), never on the angles being fitted.
    """
    q, Q = sample.a * (1 - sample.e), sample.a * (1 + sample.e)
    return (Q < inner * a_planet) | (q > outer * a_planet)


def _subset(sample: Sample, mask):
    names = tuple(np.asarray(sample.names)[mask]) if sample.names else ()
    return Sample(sample.poles[mask], sample.a[mask], sample.e[mask], names)


def profile_grid(sample: Sample, masses, a_planets, ecc="quadrupole", e_planet=0.0, i_fixed=None,
                 valid_only=True, min_objects=5):
    """Profile likelihood over (mass, a_p), maximised over (i_p, Omega_p, kappa) at each point.

    With ``i_fixed`` (deg) the planet's inclination is held fixed and only
    (Omega_p, kappa) are profiled. Masses should be increasing. With
    ``valid_only`` each a_p uses only the objects in ``valid_mask``; rows with
    fewer than ``min_objects`` are left as nan.

    Returns a dict of 2-D arrays shaped (len(a_planets), len(masses)): ``loglike``,
    ``i_p``, ``Omega_p``, ``kappa``; and per-a_p arrays ``null_loglike``, ``n``.
    """
    shape = (len(a_planets), len(masses))
    out = {k: np.full(shape, np.nan) for k in ("loglike", "i_p", "Omega_p", "kappa")}
    null_ll, n_used = np.full(len(a_planets), np.nan), np.zeros(len(a_planets), int)
    for ia, ap in enumerate(a_planets):
        sub = _subset(sample, valid_mask(sample, ap)) if valid_only else sample
        n_used[ia] = len(sub)
        if len(sub) < min_objects:
            continue
        null_ll[ia] = fit_null(sub)["loglike"]
        model = PlanetModel(sub, ecc, e_planet)
        starts = _orientation_starts(sub)
        if i_fixed is not None:
            starts += [(i_fixed, o) for o in (0.0, 90.0, 180.0, 270.0)]
        prev = None
        for im, m in enumerate(masses):
            w = model.planet_weights(m, ap)
            r = _fit_orientation(model, w, starts + ([prev] if prev else []), i_fixed)
            prev = (r["i_p"], r["Omega_p"])
            for k in out:
                out[k][ia, im] = r[k]
    out.update(masses=np.asarray(masses), a_planets=np.asarray(a_planets), null_loglike=null_ll, n=n_used)
    return out


def upper_limits(grid, delta=1.35):
    """Mass upper limit (Earth masses) at each a_p, where 2 dlnL from the profile maximum
    reaches ``2 * delta`` (delta = 1.35 is the one-sided 95% limit for one parameter).

    The profile at each a_p is taken over mass on the grid, above the best-fitting mass;
    ``inf`` means the limit lies beyond the largest grid mass.
    """
    ll, ms = grid["loglike"], grid["masses"]
    lims = np.full(ll.shape[0], np.inf)
    for ia in range(ll.shape[0]):
        row = ll[ia]
        if np.all(np.isnan(row)):
            lims[ia] = np.nan
            continue
        # the m -> 0 limit is the null model, so it belongs to the profile too
        k = int(np.nanargmax(row))
        best = max(row[k], grid["null_loglike"][ia])
        k = k if row[k] >= grid["null_loglike"][ia] else -1
        drop = best - row
        above = np.where((np.arange(len(ms)) > k) & (drop >= delta))[0]
        if len(above):
            j = above[0]
            if j == 0:
                lims[ia] = float(ms[0])  # excluded already at the smallest grid mass
                continue
            # log-linear interpolation between grid masses
            x0, x1, y0, y1 = np.log(ms[j - 1]), np.log(ms[j]), drop[j - 1], drop[j]
            lims[ia] = float(np.exp(x0 + (delta - y0) * (x1 - x0) / (y1 - y0)))
    return lims


def lower_limits(grid, delta=1.35):
    """Smallest mass (Earth masses) at each a_p still within ``delta`` in lnL of the profile maximum.

    Below it the planet is too weak to tilt the poles as far as the data prefer.
    ``nan`` when the null itself is within ``delta`` of the maximum (no lower limit).
    """
    ll, ms = grid["loglike"], grid["masses"]
    lims = np.full(ll.shape[0], np.nan)
    for ia in range(ll.shape[0]):
        row = ll[ia]
        if np.all(np.isnan(row)):
            continue
        best = np.nanmax(row)
        if best - grid["null_loglike"][ia] < delta:
            continue
        lims[ia] = float(ms[np.where(best - row < delta)[0][0]])
    return lims


# ---------------------------------------------------------------- calibration

def sample_vmf(mu, kappa, n, rng):
    """Draw ``n`` unit vectors from a von Mises-Fisher density about ``mu`` (Wood 1994)."""
    mu = np.asarray(mu, float) / np.linalg.norm(mu)
    u = rng.uniform(size=n)
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(np.clip(1 - w**2, 0, None))
    v = np.stack([s * np.cos(phi), s * np.sin(phi), w], axis=-1)
    # rotate z -> mu
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, mu)
    sa, ca = np.linalg.norm(axis), mu @ z
    if sa < 1e-12:
        return v if ca > 0 else -v
    k = axis / sa
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + sa * K + (1 - ca) * K @ K
    return v @ R.T


def mock_null(sample: Sample, kappa, rng):
    """A mock sample with the same (a, e) and poles drawn about the invariable pole."""
    return Sample(sample_vmf([0, 0, 1], kappa, len(sample), rng), sample.a, sample.e)


def test_statistic(grid):
    """max over a_p of 2 [max_m lnL(planet) - lnL(null)], each on that a_p's own subset."""
    return float(np.nanmax(2.0 * (np.nanmax(grid["loglike"], axis=1) - grid["null_loglike"])))


def null_distribution(sample: Sample, masses, a_planets, n_mock=50, seed=0, **kw):
    """Test statistics of ``n_mock`` no-planet mocks, for a p-value of the observed one."""
    rng = np.random.default_rng(seed)
    kappa = fit_null(sample)["kappa"]
    return np.array([test_statistic(profile_grid(mock_null(sample, kappa, rng), masses, a_planets, **kw))
                     for _ in range(n_mock)])


def crossing_fraction(sample: Sample, a_planet):
    """Fraction of objects outside ``valid_mask``: those that approach the planet's orbit."""
    return float(np.mean(~valid_mask(sample, a_planet)))


# ---------------------------------------------------------------- the original script 25

def _script25_model(a, inc_p, mass_earth):
    """Original script 25 'secular_model', reproduced verbatim (arbitrary 1e8 / 1e10 constants)."""
    planet_m = mass_earth * 3e-6
    torque_giants = 1.0 / (a**3.5) * 1e8
    torque_p9 = planet_m * (a**2) / (500.0**3) * 1e10
    return (torque_p9 * inc_p) / (torque_giants + torque_p9)


def script25_fit(a, i):
    """Run the original curve fit on (a, i) arrays. Returns (planet inclination, mass)."""
    from scipy.optimize import curve_fit

    popt, _ = curve_fit(_script25_model, a, i, p0=[20.0, 5.0], bounds=([0, 0.1], [90, 20]))
    return float(popt[0]), float(popt[1])


def circular_fit_demo(df, n_mock=200, seed=0, i_window=(14.0, 22.0), a_min=250.0, q_min=40.0):
    """Show that script 25's selection decides its answer.

    Applies script 25's cuts (a > 250, q > 40, 14 < i < 22) and fit to the real
    sample and to ``n_mock`` mocks with the same (a, e) but isotropic nodes and
    inclinations redrawn from the real sample's own distribution (no plane at all).
    Returns dict with the real fit and the mock fit distribution.
    """
    rng = np.random.default_rng(seed)
    base = df[(df["a"] > a_min) & (df["q"] > q_min)]

    def run(a, i):
        sel = (i > i_window[0]) & (i < i_window[1])
        if sel.sum() < 4:
            return np.nan, np.nan, int(sel.sum())
        return (*script25_fit(a[sel], i[sel]), int(sel.sum()))

    real = run(base["a"].to_numpy(), base["i"].to_numpy())
    pool = df["i"].to_numpy()
    mocks = np.array([run(base["a"].to_numpy(), rng.choice(pool, len(base))) for _ in range(n_mock)])
    return {"real": real, "mock_inclination": mocks[:, 0], "mock_mass": mocks[:, 1], "mock_n": mocks[:, 2]}
