import numpy as np
import pytest

from planetix.stats import circular as C


def test_descriptives_on_known_angles():
    th = [350.0, 10.0]
    assert C.mean_direction(th) == pytest.approx(0.0, abs=1e-9) or C.mean_direction(th) == pytest.approx(360.0)
    assert C.mean_resultant_length(th) == pytest.approx(np.cos(np.deg2rad(10)))
    assert C.mean_resultant_length([0, 90, 180, 270]) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(C.mean_direction([0, 180]))


def test_rotation_invariance():
    rng = np.random.default_rng(0)
    th = rng.vonmises(1.0, 1.5, 25) * 180 / np.pi
    for f in (C.rayleigh_test, C.kuiper_test):
        a, b = f(th).p_value, f(th + 123.4).p_value
        assert a == pytest.approx(b, rel=1e-9)


@pytest.mark.parametrize("n", [8, 15, 40])
def test_rayleigh_analytic_matches_monte_carlo(n):
    rng = np.random.default_rng(n)
    null = rng.uniform(0, 2 * np.pi, (200000, n))
    Rb = np.hypot(np.cos(null).mean(1), np.sin(null).mean(1))
    for q in (0.9, 0.99):
        th = np.zeros(n)
        # build a sample whose R-bar equals the q-quantile of the null
        r = np.quantile(Rb, q)
        half = np.arccos(r)
        th = np.rad2deg(np.where(np.arange(n) % 2, half, -half))
        assert C.rayleigh_test(th).p_value == pytest.approx(1 - q, rel=0.08)


@pytest.mark.parametrize("test", [C.rayleigh_test, C.kuiper_test])
def test_uniform_size_at_small_n(test):
    rng = np.random.default_rng(1)
    p = np.array([test(rng.uniform(0, 360, 12)).p_value for _ in range(4000)])
    assert 0.035 < np.mean(p < 0.05) < 0.065


def test_kuiper_sees_bimodal_rayleigh_does_not():
    rng = np.random.default_rng(2)
    th = np.r_[rng.vonmises(0, 8, 30), rng.vonmises(np.pi, 8, 30)] * 180 / np.pi
    assert C.kuiper_test(th).p_value < 0.01
    assert C.rayleigh_test(th).p_value > 0.01


def test_kappa_bias_correction_shrinks_small_n():
    rng = np.random.default_rng(3)
    raw = [C.kappa_mle(rng.vonmises(0, 1, 10) * 57.29578, bias_correct=False) for _ in range(500)]
    rng = np.random.default_rng(3)
    cor = [C.kappa_mle(rng.vonmises(0, 1, 10) * 57.29578) for _ in range(500)]
    assert abs(np.median(cor) - 1) < abs(np.median(raw) - 1)


def test_vonmises_logpdf_normalised():
    x = np.linspace(0, 360, 20001)[:-1]
    for k in (0.0, 1.0, 50.0):
        dens = np.exp(C.vonmises_logpdf(x, 30, k))
        assert dens.mean() * 2 * np.pi == pytest.approx(1.0, rel=1e-6)


def test_circular_correlation():
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 360, 30)
    assert C.circular_correlation(a, a + rng.normal(0, 10, 30), n_perm=999, rng=0).p_value < 0.01
    assert C.circular_correlation(a, rng.uniform(0, 360, 30), n_perm=999, rng=0).p_value > 0.01


def test_bootstrap_ci_covers_and_wraps():
    rng = np.random.default_rng(5)
    th = np.rad2deg(rng.vonmises(0, 4, 30))
    ci = C.bootstrap_mean_direction(th, n_boot=2000, rng=0)
    assert ci.lower > 180 and ci.upper < 180  # interval crosses 0 deg


def test_pole_round_trip_and_perihelion_unit():
    i, O, w = np.array([20.0, 100.0]), np.array([113.0, 300.0]), np.array([150.0, 10.0])
    h = C.orbit_pole(i, O)
    for k in range(2):
        assert C.pole_to_elements(h[k]) == pytest.approx((i[k], O[k]))
    p = C.perihelion_direction(i, O, w)
    assert np.allclose(np.linalg.norm(p, axis=1), 1)
    assert np.allclose(np.sum(p * h, axis=1), 0)  # perihelion lies in the orbit plane
    # for i = 0 the perihelion longitude is Omega + omega
    p0 = C.perihelion_direction([0.0], [40.0], [30.0])[0]
    assert np.rad2deg(np.arctan2(p0[1], p0[0])) == pytest.approx(70.0)


def test_spherical_rayleigh_size():
    rng = np.random.default_rng(6)
    p = []
    for _ in range(2000):
        v = rng.normal(size=(15, 3))
        p.append(C.spherical_rayleigh(v).p_value)
    assert 0.03 < np.mean(np.array(p) < 0.05) < 0.07


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        C.rayleigh_test([])
    with pytest.raises(ValueError):
        C.rayleigh_test([1.0, np.nan])


def test_vectorised_statistics_match_scalar():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 360, (5, 17))
    assert np.allclose(C.resultant_length(x), [C.mean_resultant_length(r) for r in x])
    assert np.allclose(C.kuiper_statistic(x), [C.kuiper_test(r).V for r in x])
    assert np.allclose(C.kuiper_statistic(x.T, axis=0), C.kuiper_statistic(x))
