import numpy as np
import pandas as pd
import pytest

from planetix.bias.footprint import EmpiricalFootprint, designation_date_jd
from planetix.bias.selection import clone_weights, mean_anomaly_window, selection_grid, selection_pool
from planetix.data.mpc import _jd_from_calendar


def test_designation_half_month():
    assert designation_date_jd("2012 VP113") == _jd_from_calendar(2012, 11, 8.0)  # V = Nov 1-15
    assert designation_date_jd("2015 TG387") == _jd_from_calendar(2015, 10, 8.0)  # T = Oct 1-15
    assert designation_date_jd("2003 VB12") == _jd_from_calendar(2003, 11, 8.0)
    assert designation_date_jd("2004 BB") == _jd_from_calendar(2004, 1, 23.0)  # B = Jan 16-31
    assert np.isnan(designation_date_jd("Sedna"))


def _toy_footprint(**kw):
    """Discoveries only in a patch around (lon, lat) = (30, 0) at V = 23."""
    rng = np.random.default_rng(0)
    n = 400
    ref = pd.DataFrame({
        "designation": [f"x{k}" for k in range(n)], "name": [f"x{k}" for k in range(n)],
        "q": 40.0, "a": 45.0, "disc_lon": rng.uniform(25, 35, n), "disc_lat": rng.uniform(-5, 5, n),
        "disc_V": np.full(n, 23.0),
    })
    return EmpiricalFootprint.from_catalog(discoveries=ref, **kw)


def test_weights_follow_position_and_depth():
    fp = _toy_footprint()
    assert fp.weight(30, 0, 22.0) > 0
    assert fp.weight(30, 0, 23.5) == 0  # fainter than every discovery there
    assert fp.weight(200, 0, 20.0) == 0  # nobody looked there
    assert fp.weight(30, 0, 18.0) >= fp.weight(30, 0, 22.9)


def test_exclude_removes_objects():
    fp_all = _toy_footprint()
    fp_ex = _toy_footprint(exclude=["x0", "x1", "x2"])
    assert len(fp_ex.reference) == len(fp_all.reference) - 3


def test_mean_anomaly_window():
    assert mean_anomaly_window(45.0, 0.05, 5.0, 27.0) == 180.0  # always bright enough
    assert mean_anomaly_window(500.0, 0.9, 12.0, 20.0) == 0.0  # never
    w = mean_anomaly_window(500.0, 0.9, 6.0, 25.0)
    assert 0 < w < 5


def test_selection_concentrates_on_footprint():
    """With all effort in one patch, detected clones of a high-e orbit have perihelia near it."""
    fp = _toy_footprint()
    obj = {"a": 300.0, "e": 0.85, "i": 2.0, "H": 6.0}
    pool = selection_pool(obj, fp, n_pool=2000, rng=1)
    varpi = np.mod(pool["Omega"] + pool["omega"], 360)
    mean_dir = np.degrees(np.angle(np.exp(1j * np.radians(varpi)).mean())) % 360
    assert abs((mean_dir - 30 + 180) % 360 - 180) < 25
    g = selection_grid(obj, fp, bins=(72, 72), rng=2)
    assert g.shape == (72, 72) and g.sum() == pytest.approx(1.0)


def test_clone_weights_columns():
    c = clone_weights({"a": 300.0, "e": 0.85, "i": 10.0, "H": 6.0}, _toy_footprint(), 1000, rng=0)
    assert list(c.columns) == ["Omega", "omega", "varpi", "M", "weight"]
    assert (c["weight"] >= 0).all()
