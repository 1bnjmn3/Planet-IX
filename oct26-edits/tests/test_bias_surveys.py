import numpy as np
import pytest

from planetix.bias.surveys import SURVEY_DIR, SURVEYS, Efficiency, Pointing, SurveySet, read_pointings


def test_square_efficiency_matches_file():
    eff = Efficiency.read(SURVEY_DIR / "OSSOS_v11" / "2013AE.eff")
    assert eff.filter == "r"
    # 2013AE: square_param 0.8877 0.0276 24.142 0.1537 for rates 0.5-8 "/hr, mag_lim 24.09
    m = 23.0
    expected = (0.887741923 - 2.76305359e-02 * (m - 21) ** 2) / (1 + np.exp((m - 24.1423416) / 0.153656587))
    assert eff(m, 3.0) == pytest.approx(expected, rel=1e-6)
    assert eff(24.5, 3.0) == 0.0  # beyond the characterisation limit
    assert eff(23.0, 0.1) == 0.0  # below the rate cut


def test_all_surveys_parse():
    for name in SURVEYS:
        pts = read_pointings(SURVEY_DIR / name)
        assert pts and all(len(p.ra) >= 4 for p in pts)


def test_polygon_contains_centre_not_far():
    p = Pointing("t", np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0]), 0.0, 0.0, 1.0, None)
    assert p.contains(np.array([0.0, 359.5, 0.5]), np.array([0.0, 0.5, -0.5])).all()
    assert not p.contains(np.array([2.0, 180.0]), np.array([0.0, 0.0])).any()


def test_real_detection_is_detectable():
    """The published orbit of an OSSOS discovery is detectable at its discovery epoch."""
    ss = SurveySet.load(["OSSOS_v11"])
    det = ss.detections()
    o = det[det["object"] == "o5p060"].iloc[0]
    from planetix.bias.orbits import mean_motion

    M = mean_motion(o.a) * (o.JD - (o.tperi + 2400000.5))
    p = ss.detection_probability(o.a, o.e, o.i, o.Hsur, [o.Omega], [o.omega], [M], epoch_jd=o.JD, colour={"r": 0.0})
    assert p[0] > 0.3
