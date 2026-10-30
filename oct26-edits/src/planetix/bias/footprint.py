"""Empirical survey footprint from where and how faint every known distant object was found.

This follows the idea of Brown (2017, AJ 154, 65): no list of all survey pointings exists,
but the discovery positions and magnitudes of all known Kuiper-belt objects trace where
surveys have looked and how deep they went. A simulated object at sky position (lon, lat)
with magnitude V is given a detection weight equal to the number of known objects
discovered within ``radius`` degrees of that position at magnitude V or fainter, i.e. the
number of discoveries made by searches that were deep enough to see it there.

Two approximations, both inherited from Brown (2017), are worth keeping in mind:

* The density of discoveries mixes survey effort with the real density of objects
  (the classical belt and Plutinos are concentrated near the ecliptic and, for
  Plutinos, away from Neptune). Objects in the tested sample are removed from the
  reference set, and ``exclude_resonant`` drops the strongly clumped Plutinos.
* Discovery dates come from the provisional designation (exact to the half month).
  Named objects without one fall back to the first observation implied by the
  observed arc, which can be a precovery decades earlier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree

from planetix.bias.orbits import geocentric, mean_motion
from planetix.data.catalog import DEFAULT_SNAPSHOT
from planetix.data.mpc import _jd_from_calendar, read_mpcorb

# Provisional designation "2012 VP113": year, half-month letter (I skipped), order letter.
_PROVISIONAL = re.compile(r"^(\d{4}) ([A-HJ-Y])[A-HJ-Z]\d*$")
_HALF_MONTHS = "ABCDEFGHJKLMNOPQRSTUVWXY"


def designation_date_jd(designation: str) -> float:
    """Julian date of the middle of the half month encoded in a provisional designation.

    ``"2012 VP113"`` -> 2012 November 8 (V = Nov 1-15). Returns NaN if the string is not
    a modern provisional designation (e.g. a name such as ``"Sedna"``).
    """
    m = _PROVISIONAL.match(str(designation).strip())
    if not m:
        return np.nan
    k = _HALF_MONTHS.index(m.group(2))
    return _jd_from_calendar(int(m.group(1)), k // 2 + 1, 8.0 if k % 2 == 0 else 23.0)


def _yyyymmdd_to_jd(s) -> float:
    s = str(s).strip()
    if not re.fullmatch(r"\d{8}", s):
        return np.nan
    return _jd_from_calendar(int(s[:4]), int(s[4:6]), float(s[6:8]))


def discovery_table(path=None) -> pd.DataFrame:
    """Every orbit in an MPCORB file with its estimated discovery date, sky position and V.

    Adds columns ``disc_jd``, ``disc_source`` ("designation" or "arc"), ``disc_lon``,
    ``disc_lat`` (geocentric ecliptic, deg), ``disc_r``, ``disc_delta`` (AU) and ``disc_V``.
    """
    df = read_mpcorb(path or DEFAULT_SNAPSHOT)
    jd_desig = df["name"].map(designation_date_jd)
    jd_arc = df["last_obs"].map(_yyyymmdd_to_jd) - df["arc_days"]
    df["disc_jd"] = jd_desig.fillna(jd_arc)
    df["disc_source"] = np.where(jd_desig.notna(), "designation", "arc")
    df = df[df["disc_jd"].notna() & (df["e"] < 1) & (df["a"] > 0)].copy()
    M = df["M"] + mean_motion(df["a"]) * (df["disc_jd"] - df["epoch_jd"])
    lon, lat, r, delta = geocentric(df["a"], df["e"], df["i"], df["Omega"], df["omega"], M, df["disc_jd"])
    df["disc_lon"], df["disc_lat"], df["disc_r"], df["disc_delta"] = lon, lat, r, delta
    df["disc_V"] = df["H"] + 5 * np.log10(r * delta)
    return df.reset_index(drop=True)


def _lonlat_to_unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


@dataclass
class EmpiricalFootprint:
    """Smoothed counts of discoveries per sky cell and limiting magnitude.

    ``counts[c, k]`` is the number of reference discoveries within ``radius`` deg of the
    centre of cell ``c`` with discovery magnitude >= ``mag_edges[k]``.
    """

    counts: np.ndarray
    mag_edges: np.ndarray
    cell_deg: float
    radius: float
    weighting: str
    reference: pd.DataFrame = field(repr=False)

    @classmethod
    def from_catalog(
        cls,
        exclude=(),
        q_min: float = 30.0,
        radius: float = 5.0,
        cell_deg: float = 2.0,
        mag_edges=None,
        weighting: str = "count",
        exclude_resonant: bool = False,
        discoveries: pd.DataFrame | None = None,
    ) -> "EmpiricalFootprint":
        """Build the footprint from the bundled MPC snapshot.

        Parameters
        ----------
        exclude
            Designations or names to leave out of the reference set (normally the
            objects whose selection is being estimated, to avoid circularity).
        q_min
            Only objects with perihelion beyond ``q_min`` AU trace the TNO surveys.
        radius
            Smoothing radius in degrees.
        weighting
            ``"count"``: weight = number of qualifying discoveries nearby.
            ``"binary"``: weight = 1 if there is at least one, else 0.
        exclude_resonant
            Drop Plutinos (39.0 < a < 40.0 AU), whose sky positions cluster 90 deg from
            Neptune and so trace the population, not the surveys.
        discoveries
            Precomputed ``discovery_table()`` output (saves re-reading the file).
        """
        if weighting not in ("count", "binary"):
            raise ValueError("weighting must be 'count' or 'binary'")
        disc = discovery_table() if discoveries is None else discoveries
        excl = {str(x) for x in exclude}
        keep = (disc["q"] > q_min) & ~disc["designation"].astype(str).isin(excl) & ~disc["name"].astype(str).isin(excl)
        if exclude_resonant:
            keep &= ~disc["a"].between(39.0, 40.0)
        ref = disc[keep & np.isfinite(disc["disc_V"])].reset_index(drop=True)
        mag_edges = np.arange(14.0, 27.01, 0.25) if mag_edges is None else np.asarray(mag_edges, float)

        nlon, nlat = int(round(360 / cell_deg)), int(round(180 / cell_deg))
        ilon = np.clip((ref["disc_lon"].to_numpy() // cell_deg).astype(int), 0, nlon - 1)
        ilat = np.clip(((ref["disc_lat"].to_numpy() + 90) // cell_deg).astype(int), 0, nlat - 1)
        cell = ilat * nlon + ilon
        # Raw cumulative counts: discoveries in each cell at or fainter than each edge.
        kmag = np.searchsorted(mag_edges, ref["disc_V"].to_numpy(), side="right") - 1
        raw = np.zeros((nlon * nlat, len(mag_edges)))
        # Faint outliers are bad orbits (e -> 1), not real 30th-magnitude discoveries.
        valid = (kmag >= 0) & (ref["disc_V"].to_numpy() <= mag_edges[-1] + 1.0)
        np.add.at(raw, (cell[valid], np.clip(kmag[valid], 0, len(mag_edges) - 1)), 1.0)
        raw = np.cumsum(raw[:, ::-1], axis=1)[:, ::-1]
        counts = _smoothing_matrix(nlon, nlat, cell_deg, radius) @ raw
        return cls(counts, mag_edges, cell_deg, radius, weighting, ref)

    @property
    def faintest_V(self) -> float:
        """No clone fainter than this can get a non-zero weight."""
        return float(self.mag_edges[-1])

    def _cell(self, lon, lat):
        nlon, nlat = int(round(360 / self.cell_deg)), int(round(180 / self.cell_deg))
        ilon = np.clip((np.mod(lon, 360) // self.cell_deg).astype(int), 0, nlon - 1)
        ilat = np.clip(((np.asarray(lat) + 90) // self.cell_deg).astype(int), 0, nlat - 1)
        return ilat * nlon + ilon

    def weight(self, lon, lat, V):
        """Detection weight for objects at ecliptic (lon, lat) with magnitude V."""
        lon, lat, V = np.broadcast_arrays(np.asarray(lon, float), np.asarray(lat, float), np.asarray(V, float))
        k = np.searchsorted(self.mag_edges, V, side="left")
        out = np.zeros(V.shape)
        ok = k < len(self.mag_edges)  # fainter than every reference discovery: undetectable
        k = np.clip(k, 0, len(self.mag_edges) - 1)
        w = self.counts[self._cell(lon, lat), k]
        out[ok] = w[ok]
        return (out > 0).astype(float) if self.weighting == "binary" else out


def _smoothing_matrix(nlon, nlat, cell_deg, radius):
    """Sparse cell x cell matrix summing every cell whose centre lies within ``radius``."""
    lon_c = (np.arange(nlon) + 0.5) * cell_deg
    lat_c = (np.arange(nlat) + 0.5) * cell_deg - 90
    LON, LAT = np.meshgrid(lon_c, lat_c)
    xyz = _lonlat_to_unit(LON.ravel(), LAT.ravel())
    tree = BallTree(xyz)
    chord = 2 * np.sin(np.radians(max(radius, cell_deg / 2)) / 2)
    nbrs = tree.query_radius(xyz, chord)
    rows = np.repeat(np.arange(len(nbrs)), [len(n) for n in nbrs])
    cols = np.concatenate(nbrs)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(xyz), len(xyz)))
