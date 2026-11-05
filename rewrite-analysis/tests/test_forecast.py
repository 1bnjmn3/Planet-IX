import numpy as np
import pytest

from planetix.inference.lsst import LSSTFootprint, information_rate, n_required
from planetix.inference.model import PARAMS, Likelihood


def test_lsst_footprint_area():
    fp = LSSTFootprint()
    lon, lat = np.meshgrid(np.arange(0.5, 360, 1.0), np.arange(-89.5, 90, 1.0), indexing="ij")
    area = (fp.covered(lon, lat) * np.cos(np.radians(lat))).sum() * (np.pi / 180) ** 2 * (180 / np.pi) ** 2
    assert 17_000 < area < 22_000  # WFD + NES, galactic plane removed


def test_information_rate_zero_without_planet_and_positive_with():
    rng = np.random.default_rng(0)
    n, nb = 5, 24
    S = rng.random((n, nb, nb))
    S /= S.sum(axis=(1, 2), keepdims=True)
    like = Likelihood(rng.uniform(0, 360, n), rng.uniform(0, 360, n), rng.uniform(5, 30, n),
                      np.full(n, 600.0), S)
    theta = [np.log10(6.0), np.log10(400.0), 0.2, 20.0, 247.0, 96.0, 0.0, 0.5, 1.3]
    assert np.allclose(information_rate(like, theta), 0.0, atol=1e-12)
    theta[PARAMS.index("f_max")] = 0.8
    assert np.all(information_rate(like, theta) > 0)


def test_n_required_monotone():
    assert n_required(0.3, np.log(100), 2.0, 0.0) < n_required(0.15, np.log(100), 2.0, 0.0)


def test_population_elements_roundtrip():
    rebound = pytest.importorskip("rebound")
    from planetix.dynamics.constants import GM_SUN
    from planetix.dynamics.population import _elements

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0)
    sim.add(a=300.0, e=0.85, inc=0.4, Omega=1.0, omega=2.0, M=0.5)
    p = sim.particles[1]
    el, _ = _elements(np.array([p.xyz]), np.array([p.vxyz]), GM_SUN)
    assert np.allclose(el[0], [300.0, 0.85, *np.degrees([0.4, 1.0, 2.0, 0.5])], rtol=1e-8)
