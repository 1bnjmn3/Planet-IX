import numpy as np
import pytest

from planetix.stats import nulls as N
from planetix.stats.circular import rayleigh_test


def test_mc_p_value_never_zero():
    assert N.mc_p_value(10.0, np.zeros(99)) == pytest.approx(0.01)
    assert N.mc_p_value(-1.0, np.zeros(99)) == 1.0


def test_rayleigh_mc_agrees_with_analytic():
    rng = np.random.default_rng(0)
    th = np.rad2deg(rng.vonmises(0, 0.6, 20))
    mc = N.rayleigh_mc(th, n_sims=50000, rng=1)
    assert mc.p_ci95[0] - 0.01 < rayleigh_test(th).p_value < mc.p_ci95[1] + 0.01


def test_generic_monte_carlo_test_matches_rayleigh_mc():
    rng = np.random.default_rng(0)
    th = np.rad2deg(rng.vonmises(0, 0.6, 20))
    stat = lambda x: rayleigh_test(x).R_bar
    a = N.monte_carlo_test(th, stat, N.uniform_angles(20), n_sims=5000, rng=2)
    b = N.rayleigh_mc(th, n_sims=5000, rng=2)
    assert a.p_value == pytest.approx(b.p_value, abs=0.03)


def test_biased_reference_null_absorbs_selection():
    # Detections pile up near 0 deg purely from selection. A uniform null
    # calls that significant; the reference-population null does not.
    rng = np.random.default_rng(3)
    biased = lambda m: np.rad2deg(rng.vonmises(0, 1.0, m))
    ref, sample = biased(2000), biased(30)
    assert N.rayleigh_mc(sample, n_sims=5000, rng=0).p_value < 0.01
    assert N.rayleigh_mc(sample, N.resample_reference(ref, 30), n_sims=5000, rng=0).p_value > 0.05


def test_from_density_reproduces_shape():
    grid = np.arange(5, 360, 10.0)
    w = np.where(grid < 180, 3.0, 1.0)
    x = N.from_density(grid, w, 40000)(np.random.default_rng(0))
    assert np.mean(x < 180) == pytest.approx(0.75, abs=0.01)
    assert x.min() >= 0 and x.max() < 360


def _nested_masks(n_total):
    # nested "a > cut" style masks over objects sorted by a
    return [np.arange(n_total) >= k for k in (0, 10, 20, 30)]


def test_scan_cuts_global_p_is_calibrated():
    rng = np.random.default_rng(4)
    masks = _nested_masks(45)
    g, loc = [], []
    for _ in range(300):
        r = N.scan_cuts(rng.uniform(0, 360, 45), masks, n_sims=500, rng=rng)
        g.append(r.global_p)
        loc.append(r.best_local_p)
    g, loc = np.array(g), np.array(loc)
    assert 0.02 < np.mean(g < 0.05) < 0.09  # corrected: about nominal
    assert np.mean(loc < 0.05) > 0.08  # naive best-cut p: inflated


def test_scan_cuts_detects_signal_in_subsample():
    rng = np.random.default_rng(5)
    th = rng.uniform(0, 360, 45)
    th[30:] = np.rad2deg(rng.vonmises(1.0, 6.0, 15))
    r = N.scan_cuts(th, _nested_masks(45), labels=["all", "a1", "a2", "a3"], n_sims=4000, rng=0)
    assert r.best_label == "a3" and r.global_p < 0.01
    assert list(r.n_per_cut) == [45, 35, 25, 15]


def test_power_monotone_and_size():
    pw = N.rayleigh_power(20, [0.0, 0.5, 1.0, 2.0], n_sims=5000, rng=0)
    assert pw[0] == pytest.approx(0.05, abs=0.015)
    assert np.all(np.diff(pw) > 0)
    assert N.min_detectable_kappa(40, rng=0) < N.min_detectable_kappa(10, rng=0)
