"""Characterised surveys (OSSOS v11 release): footprints, epochs and detection efficiencies.

The OSSOS collaboration published, for each of its survey blocks and those of CFEPS, the
High Latitude survey and the MA survey, the exact sky polygons, the epoch of the discovery
images, the filling factor and an efficiency function of magnitude and rate of motion
(Bannister et al. 2018, ApJS 236, 18; Petit et al. 2011; Petit et al. 2017). Those files
are bundled in ``data/external/ossos_v11`` and read here.

``SurveySet.detection_probability`` is a light re-implementation of the OSSOS survey
simulator's detection step in numpy: propagate the orbit to each block's epoch, test
whether it falls in the polygon, compute the magnitude in the block's filter and the
on-sky rate, and apply the rate cut, efficiency, characterisation limit and tracking
fraction. Not modelled: the direction-of-motion cut, magnitude noise and the per-chip
gaps beyond the filling factor. For the slow, distant objects studied here these change
detection probabilities by a few per cent, not the footprint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from planetix.bias.orbits import geocentric, mean_motion
from planetix.data.catalog import DATA_DIR

SURVEY_DIR = DATA_DIR / "external" / "ossos_v11"
SURVEYS = ("OSSOS_v11", "CFEPS_g", "CFEPS_r", "CFEPS_presurvey", "HiLat", "MA")

# Filter minus V for an average TNO (colour.toml of the characterisation release:
# V-g = -0.42, r-g = -0.7, R-g = -0.8), so m_filter = V + offset.
COLOUR_MINUS_V = {"V": 0.0, "g": 0.42, "r": -0.28, "R": -0.38}


def _sexagesimal(s: str, hours: bool) -> float:
    s = s.strip()
    if ":" not in s:
        return float(s)
    sign = -1.0 if s.startswith("-") else 1.0
    # Take the first three fields: CFEPS_g/pointings.list has one typo, "-21:57:38:0".
    d, m, sec = (abs(float(x)) for x in s.lstrip("+-").split(":")[:3])
    val = sign * (d + m / 60 + sec / 3600)
    return val * 15.0 if hours else val


@dataclass
class Efficiency:
    """One block's efficiency file."""

    filter: str
    rate_min: float
    rate_max: float
    track: tuple  # (track_max, track_mag, track_slope)
    segments: list  # [(rate_lo, rate_hi, kind, params, mag_lim)]

    @classmethod
    def read(cls, path) -> "Efficiency":
        rate_cut, track, filt, segs, glob_lim = None, (1.0, 99.0, 0.0), "r", [], np.inf
        cur = None
        for raw in Path(path).read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, val = (x.strip() for x in line.split("=", 1))
            nums = val.split()
            if key == "rate_cut":
                rate_cut = (float(nums[0]), float(nums[1]))
            elif key == "track_frac":
                track = tuple(float(x) for x in nums[:3])
            elif key == "filter":
                filt = nums[0]
            elif key == "rates":
                cur = [float(nums[0]), float(nums[1]), None, None, None]
                segs.append(cur)
            elif key == "function":
                cur[2] = nums[0]
            elif key.endswith("_param") and key != "lookup_param":
                cur[3] = [float(x) for x in nums]
            elif key == "mag_lim":
                if cur is None:
                    glob_lim = float(nums[0])
                else:
                    cur[4] = float(nums[0])
        if rate_cut is None:
            rate_cut = (0.0, np.inf)
        segs = [(lo, hi, k, p, glob_lim if lim is None else lim) for lo, hi, k, p, lim in segs]
        if any(k not in ("single", "double", "square", "linear") for _, _, k, _, _ in segs):
            raise ValueError(f"Unsupported efficiency function in {path}")
        return cls(filt, rate_cut[0], rate_cut[1], track, segs)

    def __call__(self, mag, rate):
        """Detection x tracking efficiency for magnitudes ``mag`` and rates (arcsec/hr)."""
        mag, rate = np.broadcast_arrays(np.asarray(mag, float), np.asarray(rate, float))
        eta = np.zeros(mag.shape)
        for lo, hi, kind, p, lim in self.segments:
            sel = (rate >= lo) & (rate < hi) & (mag <= lim)
            if not sel.any():
                continue
            m = mag[sel]
            if kind == "single":
                m0 = p[1] + (p[3] * rate[sel] if len(p) > 3 else 0.0)
                val = p[0] / 2 * (1 - np.tanh((m - m0) / p[2]))
            elif kind == "double":
                val = p[0] / 4 * (1 - np.tanh((m - p[1]) / p[2])) * (1 - np.tanh((m - p[1]) / p[3]))
            elif kind == "square":
                val = np.where(m < 21.0, p[0], (p[0] - p[1] * (m - 21.0) ** 2) / (1 + np.exp((m - p[2]) / p[3])))
            else:  # linear
                val = np.where(m < p[1], p[0], np.where(m < p[2], (m - p[2]) * p[0] / (p[1] - p[2]), 0.0))
            eta[sel] = np.clip(val, 0, 1)
        eta[(rate < self.rate_min) | (rate > self.rate_max)] = 0.0
        tmax, tmag, tslope = self.track
        return eta * np.clip(np.minimum(tmax, 1 + (mag - tmag) * tslope), 0, 1)


@dataclass
class Pointing:
    survey: str
    ra: np.ndarray  # polygon vertices, deg (unwrapped around ra0)
    dec: np.ndarray
    ra0: float
    epoch: float
    fill: float
    eff: Efficiency

    def contains(self, ra, dec):
        """Point-in-polygon (ray casting) in the (RA, Dec) plane, RA wrapped about ra0."""
        x = self.ra0 + (np.asarray(ra) - self.ra0 + 180.0) % 360.0 - 180.0
        y = np.asarray(dec)
        inside = np.zeros(np.shape(x), bool)
        xv, yv = self.ra, self.dec
        for k in range(len(xv)):
            x1, y1, x2, y2 = xv[k - 1], yv[k - 1], xv[k], yv[k]
            crosses = (y1 > y) != (y2 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (x < xc)
        return inside


def read_pointings(survey_dir) -> list[Pointing]:
    """Parse a ``pointings.list`` (``poly`` and rectangle lines) and its efficiency files."""
    survey_dir = Path(survey_dir)
    lines = [l.split("#", 1)[0].strip() for l in (survey_dir / "pointings.list").read_text().splitlines()]
    lines = [l for l in lines if l]
    effs, out, j = {}, [], 0
    while j < len(lines):
        w = lines[j].split()
        j += 1
        if w[0] == "rect":
            w = w[1:]
        if w[0] == "poly":
            n = int(w[1])
            ra0, dec0 = _sexagesimal(w[2], True), _sexagesimal(w[3], False)
            epoch, fill, eff_name = float(w[4]), float(w[5]), w[-1]
            off = np.array([[float(v) for v in lines[j + k].split()[:2]] for k in range(n)])
            j += n
            dec = dec0 + off[:, 1]
            ra = ra0 + off[:, 0] / np.cos(np.radians(dec))
        else:
            wid, hgt = float(w[0]), float(w[1])
            ra0, dec0 = _sexagesimal(w[2], True), _sexagesimal(w[3], False)
            epoch, fill, eff_name = float(w[4]), float(w[5]), w[-1]
            dec = dec0 + np.array([-1, -1, 1, 1]) * hgt / 2
            ra = ra0 + np.array([-1, 1, 1, -1]) * wid / 2 / np.cos(np.radians(dec))
        if eff_name not in effs:
            effs[eff_name] = Efficiency.read(survey_dir / eff_name)
        out.append(Pointing(survey_dir.name, ra, dec, ra0, epoch, fill, effs[eff_name]))
    return out


def read_detections(survey_dir) -> pd.DataFrame:
    """Characterised detections of a survey, with its barycentric orbit fits."""
    survey_dir = Path(survey_dir)
    path = next(survey_dir.glob("*.detections"))
    header = next(l for l in path.read_text().splitlines() if l.startswith("# cl"))[1:].split()
    df = pd.read_csv(path, comment="#", sep=r"\s+", names=header)
    df["survey"] = survey_dir.name
    df["q"] = df["a"] * (1 - df["e"])
    return df


@dataclass
class SurveySet:
    pointings: list = field(repr=False)
    names: tuple = SURVEYS

    @classmethod
    def load(cls, names=SURVEYS, root=SURVEY_DIR) -> "SurveySet":
        pts = [p for n in names for p in read_pointings(Path(root) / n)]
        return cls(pts, tuple(names))

    reference_epoch: float = 2457000.5  # 2014 Dec 9, inside the 2003-2017 span of the blocks

    @property
    def time_span_days(self) -> float:
        """Largest |block epoch - reference epoch|."""
        return float(max(abs(p.epoch - self.reference_epoch) for p in self.pointings))

    @property
    def faintest_V(self) -> float:
        """Faintest V any block could characterise (largest mag_lim, bluest filter offset)."""
        lim = max(seg[4] for p in self.pointings for seg in p.eff.segments if np.isfinite(seg[4]))
        return float(lim - min(COLOUR_MINUS_V.get(p.eff.filter, 0.0) for p in self.pointings))

    def detections(self, root=SURVEY_DIR) -> pd.DataFrame:
        return pd.concat([read_detections(Path(root) / n) for n in self.names], ignore_index=True)

    def detection_probability(self, a, e, i, H, Omega, omega, M, epoch_jd=None, colour=None):
        """Probability that the characterised surveys detect each orbit.

        ``M`` is the mean anomaly at ``epoch_jd`` (default ``reference_epoch``); it is
        propagated to each block's epoch. ``H`` is in V unless ``colour`` gives a dict
        filter -> (m_filter - H) offsets. Blocks are treated as independent:
        P = 1 - prod(1 - p_block). Pointings whose epochs agree to within a day share one
        ephemeris computation (the object moves < 0.05 deg in that time).
        """
        colour = COLOUR_MINUS_V if colour is None else colour
        epoch_jd = self.reference_epoch if epoch_jd is None else epoch_jd
        Omega, omega, M = np.broadcast_arrays(*(np.asarray(x, float) for x in (Omega, omega, M)))
        n = mean_motion(a)
        miss = np.ones(Omega.shape)
        dt = 1.0 / 24.0
        groups = {}
        for p in self.pointings:
            groups.setdefault(round(p.epoch), []).append(p)
        for pts in groups.values():
            t = pts[0].epoch
            Mp = M + n * (t - epoch_jd)
            ra, dec, r, delta = geocentric(a, e, i, Omega, omega, Mp, t, frame="equatorial")
            for p in pts:
                k = np.flatnonzero(p.contains(ra, dec))
                if not len(k):
                    continue
                ra2, dec2, _, _ = geocentric(a, e, i, Omega[k], omega[k], Mp[k] + n * dt, t + dt, frame="equatorial")
                rate = _separation(ra[k], dec[k], ra2, dec2) * 3600.0  # arcsec per hour
                mag = H + colour.get(p.eff.filter, 0.0) + 5 * np.log10(r[k] * delta[k])
                miss[k] *= 1.0 - p.fill * p.eff(mag, rate)
        return 1.0 - miss


def _separation(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees (haversine)."""
    ra1, dec1, ra2, dec2 = (np.radians(x) for x in (ra1, dec1, ra2, dec2))
    h = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(h)))
