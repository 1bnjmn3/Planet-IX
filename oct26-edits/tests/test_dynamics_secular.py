import numpy as np
import pytest

from planetix.dynamics import (
    Perturber, coupling_rate, ecliptic_to_invariable, forced_inclination, forced_pole, giant_planets,
    invariable_to_ecliptic, laplace_coefficient, nodal_precession_rate, planet_nine, pole, pole_to_angles,
    required_mass, two_plane_tilt,
)
from planetix.dynamics.constants import GIANTS, giants_j2r2, mean_motion


def test_laplace_coefficient_matches_series():
    for alpha in (0.02, 0.05, 0.1):
        series = 3 * alpha * (1 + 15 / 8 * alpha**2 + 175 / 64 * alpha**4 + 3675 / 1024 * alpha**6)
        assert laplace_coefficient(1.5, 1, alpha) == pytest.approx(series, rel=1e-5)


def test_coupling_rate_limits_are_quadrupole():
    # inner perturber far inside: B = (3/4) n m (a_j/a)^2 ; outer far outside: (3/4) n m (a/a_j)^3
    p_in, p_out = Perturber(1e-3, 5.0), Perturber(1e-3, 50000.0)
    a = 500.0
    assert coupling_rate(a, p_in) == pytest.approx(0.75 * mean_motion(a) * 1e-3 * (5 / a) ** 2, rel=1e-3)
    assert coupling_rate(a, p_out) == pytest.approx(0.75 * mean_motion(a) * 1e-3 * (a / 50000) ** 3, rel=1e-3)


def test_giants_match_their_j2_far_out():
    a = 2000.0
    j2_rate = 1.5 * mean_motion(a) * giants_j2r2() / a**2
    assert -nodal_precession_rate(a, giant_planets()) == pytest.approx(j2_rate, rel=1e-3)
    assert len(GIANTS) == 4


def test_eccentricity_factors():
    a, e = 400.0, 0.8
    g = giant_planets()[0]
    assert coupling_rate(a, g, e, "quadrupole") / coupling_rate(a, g) == pytest.approx((1 - e**2) ** -2)
    p = Perturber.earths(5, 2000, e=0.3)
    ratio = (1 + 1.5 * e**2) / np.sqrt(1 - e**2) * (1 - 0.3**2) ** -1.5
    assert coupling_rate(a, p, e, "quadrupole") / coupling_rate(a, p) == pytest.approx(ratio)
    with pytest.raises(ValueError):
        coupling_rate(a, p, e, "bogus")


def test_forced_tilt_rises_from_giants_plane_to_planet_plane():
    pl = planet_nine(5.0, 500.0, 20.0, 100.0)
    a = np.array([60.0, 150.0, 250.0, 350.0, 450.0])
    i, O = forced_inclination(a, pl, method="nonlinear")
    assert np.all(np.diff(i) > 0)
    assert i[0] < 0.5 and 18.0 < i[-1] < 20.0
    assert np.allclose(O, 100.0)


def test_linear_and_nonlinear_agree_for_small_tilt():
    pl = planet_nine(5.0, 500.0, 2.0, 40.0)
    for a in (150.0, 250.0, 350.0):
        assert forced_inclination(a, pl, method="linear")[0] == pytest.approx(
            forced_inclination(a, pl, method="nonlinear")[0], abs=2e-3)


def test_two_plane_tilt_is_the_nonlinear_solution():
    P = Perturber.earths(5.0, 500.0, 60.0, 0.0)
    for a, e in [(250.0, 0.7), (300.0, 0.2)]:
        w_ref = sum(coupling_rate(a, g, e, "quadrupole") for g in giant_planets())
        w_p = coupling_rate(a, P, e, "quadrupole")
        assert two_plane_tilt(w_ref, w_p, 60.0) == pytest.approx(
            forced_inclination(a, giant_planets() + [P], e, "nonlinear", "quadrupole")[0], abs=1e-6)


def test_forced_pole_shape_and_norm():
    f = forced_pole(np.array([[200.0, 300.0]]), planet_nine())
    assert f.shape == (1, 2, 3)
    assert np.allclose(np.linalg.norm(f, axis=-1), 1.0)


def test_required_mass_inverts_forced_inclination():
    m = required_mass(10.0, 250.0, 500.0, 20.0)
    pl = giant_planets() + [Perturber.earths(m, 500.0, 20.0, 0.0)]
    assert forced_inclination(250.0, pl, method="nonlinear")[0] == pytest.approx(10.0, abs=1e-3)
    assert np.isnan(required_mass(25.0, 250.0, 500.0, 20.0))  # tilt cannot exceed the planet's


def test_frames_round_trip_and_invariable_pole():
    i, O = ecliptic_to_invariable(1.57869, 107.58232)
    assert i == pytest.approx(0.0, abs=1e-9)
    i2, O2 = invariable_to_ecliptic(*ecliptic_to_invariable(23.0, 211.0))
    assert (i2, O2) == (pytest.approx(23.0), pytest.approx(211.0))
    assert np.allclose(pole_to_angles(pole(30.0, 300.0)), (30.0, 300.0))
