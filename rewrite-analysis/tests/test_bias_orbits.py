import numpy as np
import pytest

from planetix.bias.orbits import (
    earth_xyz, geocentric, heliocentric_xyz, mean_motion, perihelion_unit_vector, pole_unit_vector, solve_kepler,
)
from planetix.bias.sky import galactic_latitude, galactic_plane_ecliptic
from planetix.bias.surveys import SURVEY_DIR, read_detections


def test_kepler_solves_high_eccentricity():
    M = np.linspace(0, 2 * np.pi, 1001)
    for e in (0.0, 0.5, 0.9, 0.99):
        E = solve_kepler(M, e)
        assert np.allclose(E - e * np.sin(E), np.mod(M, 2 * np.pi), atol=1e-10)


def test_perihelion_position_and_distance():
    xyz = heliocentric_xyz(500.0, 0.9, 20.0, 110.0, 300.0, 0.0)
    assert np.linalg.norm(xyz) == pytest.approx(50.0)
    assert np.allclose(xyz / 50.0, perihelion_unit_vector(20.0, 110.0, 300.0))


def test_pole_is_normal_to_orbit():
    xyz = heliocentric_xyz(300.0, 0.5, 35.0, 40.0, 80.0, np.linspace(0, 360, 7))
    assert np.allclose(xyz @ pole_unit_vector(35.0, 40.0), 0.0, atol=1e-9)


def test_earth_one_au_and_one_year():
    jd = 2451545.0 + np.arange(0, 365.25, 30)
    r = np.linalg.norm(earth_xyz(jd), axis=-1)
    assert np.all((r > 0.98) & (r < 1.02))
    assert np.allclose(earth_xyz(2451545.0), earth_xyz(2451545.0 + 365.25636), atol=1e-3)


def test_predicted_positions_match_ossos_astrometry():
    """Two-body positions from the published OSSOS orbits land on the reported RA/Dec."""
    det = read_detections(SURVEY_DIR / "OSSOS_v11")
    det = det[det["a"] > 0].head(200)
    M = mean_motion(det["a"]) * (det["JD"] - (det["tperi"] + 2400000.5))
    ra, dec, _, _ = geocentric(det["a"], det["e"], det["i"], det["Omega"], det["omega"], M, det["JD"], frame="equatorial")
    dra = ((ra - det["RAdeg"] + 180) % 360 - 180) * np.cos(np.radians(dec))
    sep = np.hypot(dra, dec - det["DEdeg"])
    assert np.median(sep) < 0.06 and sep.max() < 0.2


def test_galactic_plane():
    # Galactic centre is at ecliptic (266.84, -5.54); the plane peaks at beta = 60.2 deg near lambda = 0.
    assert abs(galactic_latitude(266.84, -5.54)) < 0.05
    lon, lat = galactic_plane_ecliptic()
    assert lat.max() == pytest.approx(60.19, abs=0.05)
    assert min(lon[np.argmax(lat)], 360 - lon[np.argmax(lat)]) < 1.0
