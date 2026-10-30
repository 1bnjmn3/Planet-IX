import numpy as np
import pytest

from planetix.data import load_catalog, load_etnos
from planetix.dynamics.frames import angle_between, pole
from planetix.dynamics.inversion import (
    PlanetModel, Sample, circular_fit_demo, fit_common_plane, fit_null, profile_grid, sample_vmf,
    script25_fit, upper_limits, valid_mask,
)
from planetix.dynamics.secular import planet_nine, forced_pole


def test_vmf_sampler_has_the_right_concentration():
    rng = np.random.default_rng(0)
    mu = pole(20.0, 50.0)
    v = sample_vmf(mu, 30.0, 20000, rng)
    # mean cosine of a vMF is coth(k) - 1/k
    assert np.mean(v @ mu) == pytest.approx(1 / np.tanh(30.0) - 1 / 30.0, abs=2e-3)
    s = Sample(v, np.full(len(v), 300.0), np.full(len(v), 0.5))
    cp = fit_common_plane(s)
    assert angle_between(cp["pole"], mu) < 0.5
    assert cp["kappa"] == pytest.approx(30.0, rel=0.05)


def test_planet_fit_recovers_an_injected_plane():
    rng = np.random.default_rng(1)
    a = rng.uniform(150, 350, 300)
    e = np.full_like(a, 0.3)
    truth = planet_nine(10.0, 1000.0, 25.0, 140.0)
    f = forced_pole(a, truth, e, "nonlinear", "quadrupole")
    poles = np.array([sample_vmf(fk, 200.0, 1, rng)[0] for fk in f])
    s = Sample(poles, a, e)
    g = profile_grid(s, [1.0, 10.0, 100.0], [1000.0])
    k = int(np.nanargmax(g["loglike"][0]))
    assert g["masses"][k] == 10.0
    assert g["i_p"][0, k] == pytest.approx(25.0, abs=2.0)
    assert g["Omega_p"][0, k] == pytest.approx(140.0, abs=5.0)
    assert g["loglike"][0, k] - g["null_loglike"][0] > 50


def test_upper_limits_on_null_data_are_finite_at_high_inclination():
    rng = np.random.default_rng(2)
    a = rng.uniform(150, 350, 60)
    s = Sample(sample_vmf([0, 0, 1], 20.0, 60, rng), a, np.full_like(a, 0.3))
    g = profile_grid(s, np.geomspace(0.1, 100, 13), [1000.0], i_fixed=40.0)
    ul = upper_limits(g)
    assert np.isfinite(ul[0]) and ul[0] < 100


def test_valid_mask_depends_only_on_orbit_size():
    s = Sample.from_frame(load_etnos("ts14"))
    m = valid_mask(s, 500.0)
    Q = s.a * (1 + s.e)
    assert np.all(Q[m] < 400.0)
    assert valid_mask(Sample.from_frame(load_etnos("bb16")), 500.0).sum() == 0


def test_null_fit_on_real_sample():
    s = Sample.from_frame(load_etnos("bb16"))
    assert len(s) == 31
    r = fit_null(s)
    assert 5 < r["kappa"] < 50


def test_script25_reproduces_original_and_is_set_by_its_window():
    cat = load_catalog()
    demo = circular_fit_demo(cat[(cat["a"] > 150) & (cat["q"] > 30)], n_mock=40)
    i_p, m_p, n = demo["real"]
    assert n == 8
    assert i_p == pytest.approx(17.11, abs=0.01) and m_p == pytest.approx(1.654, abs=0.01)
    # isotropic mocks land on the same inclination: the window, not the sky, sets it
    assert np.nanmedian(demo["mock_inclination"]) == pytest.approx(17.5, abs=1.5)
