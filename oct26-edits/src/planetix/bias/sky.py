"""Sky positions of the sample and the Galactic plane (replaces old script 17).

Old script 17 drew the Milky Way as ``60 sin(lon - 280)`` and the survey "windows" as
hand-placed boxes at 300-60 deg longitude. Here the Galactic plane is computed from the
IAU Galactic frame, and the sample's discovery positions come from the same discovery
table that builds the empirical footprint.
"""

from __future__ import annotations

import numpy as np

from planetix.bias.orbits import OBLIQUITY_J2000

# Equatorial (ICRS) -> Galactic rotation (Hipparcos/ESA 1997 Vol. 1, Sect. 1.5.3).
_EQ_TO_GAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [+0.4941094279, -0.4448296300, +0.7469822445],
        [-0.8676661490, -0.1980763734, +0.4559837762],
    ]
)


def _ecl_to_eq_matrix():
    e = np.radians(OBLIQUITY_J2000)
    return np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])


def galactic_latitude(lon_ecl, lat_ecl):
    """Galactic latitude (deg) of ecliptic J2000 positions (deg)."""
    lon, lat = np.radians(lon_ecl), np.radians(lat_ecl)
    v = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    g = v @ (_EQ_TO_GAL @ _ecl_to_eq_matrix()).T
    return np.degrees(np.arcsin(np.clip(g[..., 2], -1, 1)))


def galactic_plane_ecliptic(n: int = 721):
    """The Galactic equator (b = 0) as ecliptic (lon, lat) arrays, sorted by longitude."""
    l = np.radians(np.linspace(0, 360, n))
    g = np.stack([np.cos(l), np.sin(l), np.zeros_like(l)], axis=-1)
    v = g @ (_EQ_TO_GAL @ _ecl_to_eq_matrix())  # inverse rotation = transpose
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
    lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    order = np.argsort(lon)
    return lon[order], lat[order]


def plot_sky(sample_disc, footprint=None, ax=None):
    """Discovery positions of a sample over the reference discoveries and Galactic plane.

    ``sample_disc`` is a ``discovery_table()`` subset; needs matplotlib (extra ``plot``).
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(11, 5))[1]
    if footprint is not None:
        ref = footprint.reference
        ax.scatter(ref["disc_lon"], ref["disc_lat"], s=2, c="0.7", label=f"reference discoveries (N={len(ref)})")
    glon, glat = galactic_plane_ecliptic()
    ax.plot(glon, glat, ".", ms=1, c="tab:purple", label="Galactic plane")
    ax.scatter(sample_disc["disc_lon"], sample_disc["disc_lat"], s=40, c="tab:red", ec="k",
               label=f"sample at discovery (N={len(sample_disc)})")
    ax.set(xlim=(0, 360), ylim=(-90, 90), xlabel="ecliptic longitude (deg)", ylabel="ecliptic latitude (deg)")
    ax.legend(loc="lower left", fontsize=8)
    return ax
