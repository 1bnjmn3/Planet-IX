import numpy as np
import pytest

pytest.importorskip("rebound")
pytest.importorskip("reboundx")

from planetix.dynamics.frames import pole_to_angles
from planetix.dynamics.nbody import integrate, secular_period
from planetix.dynamics.secular import giant_planets, nodal_precession_rate


def test_j2_precession_matches_secular_rate():
    a, e, boost = 200.0, 0.7, 100.0
    T = secular_period(a, None, e, "quadrupole", boost=boost)
    r = integrate(a, e, 10.0, 0.0, None, T, n_out=40, boost=boost)
    _, O = pole_to_angles(r.poles[:, 0])
    rate = np.polyfit(r.times, np.unwrap(np.radians(O)), 1)[0]
    assert rate == pytest.approx(boost * nodal_precession_rate(a, giant_planets(), e, "quadrupole"), rel=0.06)
