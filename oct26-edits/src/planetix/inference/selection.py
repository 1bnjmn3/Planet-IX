"""Per-object selection functions S_j(varpi, Omega) for the likelihood.

This is a compact Brown (2017, AJ 154, 65) style empirical footprint:
every object in the MPC distant file marks a place on the sky that some survey
covered, at least down to that object's magnitude at discovery. A clone of an
ETNO, with its (a, e, i, H) held fixed and its orientation and orbital phase
changed, is weighted by how many such "survey points" lie within ``radius_deg``
and are at least as faint as the clone.

The survey-bias thread maintains the fuller version (``planetix.bias``); the
likelihood only needs a grid with the interface of ``selection_grid`` below,
so either can be passed to ``planetix.inference.model``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from planetix.data.catalog import DEFAULT_SNAPSHOT
from planetix.data.mpc import read_mpcorb
from planetix.inference.kepler import apparent_mag, mean_anomaly_at, sky_position

_JD_J2000 = 2451545.0


def _unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def discovery_jd(df: pd.DataFrame) -> np.ndarray:
    """Approximate discovery date: the provisional-designation year when the
    object has one, otherwise the start of the observed arc, never before 1992
    (pre-1992 arcs of TNOs are precoveries, not survey detections)."""
    year = pd.to_numeric(df["name"].str.extract(r"^(\d{4}) [A-Z]{2}", expand=False), errors="coerce")
    last = pd.to_datetime(df["last_obs"], format="%Y%m%d", errors="coerce")
    first = last.dt.year + last.dt.dayofyear / 365.25 - df["arc_days"] / 365.25
    year = year.fillna(first).clip(lower=1992.0) + 0.5 * year.notna()
    return (_JD_J2000 + (year - 2000.0) * 365.25).to_numpy(float)


@dataclass
class Footprint:
    """Survey points: sky unit vectors and limiting magnitudes, plus a lookup table."""

    designation: np.ndarray
    xyz: np.ndarray
    V: np.ndarray
    radius_deg: float = 2.0

    @classmethod
    def from_catalog(cls, path=None, r_min: float = 25.0, radius_deg: float = 2.0, exclude=(), a_min: float = 0.0):
        """Survey points from an MPCORB file. ``a_min`` > 50 AU drops the resonant and
        classical belts, whose longitude structure is intrinsic rather than survey-made."""
        df = read_mpcorb(path or DEFAULT_SNAPSHOT)
        df = df[(df["e"] < 1) & df["arc_days"].notna() & (df["a"] > a_min)].copy()
        jd = discovery_jd(df)
        M = mean_anomaly_at(df["M"], df["a"], df["epoch_jd"], jd)
        lon, lat, r = sky_position(df["a"], df["e"], df["i"], df["Omega"], df["omega"], M)
        V = apparent_mag(df["H"], r)
        keep = (r > r_min) & np.isfinite(V) & ~df["designation"].isin(list(exclude)).to_numpy()
        return cls(df["designation"].to_numpy()[keep], _unit(lon[keep], lat[keep]), V[keep], radius_deg)

    def __post_init__(self):
        self._tree = cKDTree(self.xyz)
        self._chord = 2 * np.sin(np.radians(self.radius_deg) / 2)
        # Depth table on a 1-degree grid: number of points within radius at least as faint as V.
        self._lon = np.arange(0.5, 360, 1.0)
        self._lat = np.arange(-89.5, 90, 1.0)
        self._Vgrid = np.arange(16.0, 27.01, 0.25)
        LON, LAT = np.meshgrid(self._lon, self._lat, indexing="ij")
        nbrs = self._tree.query_ball_point(_unit(LON.ravel(), LAT.ravel()), self._chord)
        table = np.zeros((LON.size, self._Vgrid.size))
        for k, idx in enumerate(nbrs):
            if idx:
                v = np.sort(self.V[idx])
                table[k] = len(v) - np.searchsorted(v, self._Vgrid, side="left")
        self._table = table.reshape(LON.shape + (self._Vgrid.size,))

    def weight(self, lon, lat, V):
        """Relative detectability at (lon, lat) for a body of magnitude V."""
        il = np.clip((np.asarray(lon) % 360).astype(int), 0, 359)
        ib = np.clip((np.asarray(lat) + 90).astype(int), 0, 179)
        iv = np.clip(np.ceil((np.asarray(V) - self._Vgrid[0]) / 0.25).astype(int), 0, self._Vgrid.size)
        out = np.zeros(np.shape(il))
        ok = iv < self._Vgrid.size
        out[ok] = self._table[il[ok], ib[ok], iv[ok]]
        return out


def selection_grid(obj, footprint: Footprint, bins=(72, 72), n_phase: int = 2000, rng=None) -> np.ndarray:
    """S_j on a [varpi_bin, Omega_bin] grid (bin k centred at (k + 0.5) * 360 / n), summing to 1.

    ``obj`` needs a, e, i, H. Mean anomaly is sampled on a stratified grid
    (uniform in time) so the result is deterministic up to ``rng`` jitter.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    nv, no = bins
    varpi_c = (np.arange(nv) + 0.5) * 360.0 / nv
    Omega_c = (np.arange(no) + 0.5) * 360.0 / no
    M = (np.arange(n_phase) + rng.random(n_phase)) * 360.0 / n_phase
    # Position at Omega = 0 for every omega; a rotation in Omega only shifts longitude.
    S = np.zeros((nv, no))
    for l, Om in enumerate(Omega_c):
        omega = (varpi_c - Om) % 360.0
        lon0, lat, r = sky_position(obj["a"], obj["e"], obj["i"], 0.0, omega[:, None], M[None, :])
        V = apparent_mag(obj["H"], r)
        S[:, l] = footprint.weight(lon0 + Om, lat, V).mean(axis=1)
    total = S.sum()
    return S / total if total > 0 else np.full(bins, 1.0 / (nv * no))
