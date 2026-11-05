import numpy as np
import pandas as pd
import pytest

from planetix.inference.evidence import draw_orientations, planet_density_grid
from planetix.inference.kepler import apparent_mag, sky_position, solve_kepler
from planetix.inference.model import PARAMS, Likelihood, Prior, a_crit
from planetix.inference.nondetection import NonDetection, planet_H

THETA = [np.log10(6.0), np.log10(400.0), 0.25, 20.0, 250.0, 100.0, 0.7, np.log10(4.0), np.log10(20.0)]


def _toy(n=12, seed=0, nv=36, no=36):
    rng = np.random.default_rng(seed)
    S = rng.random((n, nv, no)) + 0.1
    S /= S.sum(axis=(1, 2), keepdims=True)
    df = pd.DataFrame({
        "varpi": rng.uniform(0, 360, n), "Omega": rng.uniform(0, 360, n),
        "i": rng.uniform(5, 40, n), "a": rng.uniform(200, 800, n),
    })
    return df, S


def test_kepler_roundtrip():
    M = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    E = solve_kepler(M, 0.9)
    assert np.allclose(E - 0.9 * np.sin(E), M, atol=1e-9)


def test_sky_position_perihelion():
    # At M = 0 the body is at perihelion, at longitude Omega + omega for i = 0.
    lon, lat, r = sky_position(300.0, 0.9, 0.0, 40.0, 30.0, 0.0)
    assert r == pytest.approx(30.0)
    assert lon == pytest.approx(70.0)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert apparent_mag(5.0, 30.0) == pytest.approx(5 + 5 * np.log10(30 * 29))


def test_a_crit_reference_and_scaling():
    assert a_crit(5.0, 500.0, 0.25) == pytest.approx(250.0)
    # heavier planet confines closer in; a more distant one further out
    assert a_crit(10.0, 500.0, 0.25) < 250.0 < a_crit(5.0, 700.0, 0.25)


def test_no_confined_population_gives_unit_bayes_factor():
    df, S = _toy()
    theta = list(THETA)
    theta[PARAMS.index("f_max")] = 0.0
    assert Likelihood.from_frame(df, S)(theta) == pytest.approx(0.0)


def test_likelihood_ratio_is_normalised_under_selection():
    # The per-object ratio is a density relative to the H0 predictive S_j, so its
    # S_j-weighted mean over all orientations must be exactly 1.
    df, S = _toy(n=3)
    like = Likelihood.from_frame(df, S)
    g = planet_density_grid(like, THETA)
    _, nv, no = S.shape
    vc = (np.arange(nv) + 0.5) * 360 / nv
    oc = (np.arange(no) + 0.5) * 360 / no
    for j in range(len(df)):
        V, O = np.meshgrid(vc, oc, indexing="ij")
        grid = pd.DataFrame({"varpi": V.ravel(), "Omega": O.ravel(), "i": df["i"][j], "a": df["a"][j]})
        Sj = np.repeat(S[j:j + 1], len(grid), axis=0)
        ratio = np.exp(Likelihood.from_frame(grid, Sj).per_object(THETA)).reshape(nv, no)
        assert np.sum(S[j] * ratio) == pytest.approx(1.0, rel=1e-9)
        assert np.allclose(ratio, g[j] / np.sum(g[j] * S[j]))


def test_draw_orientations_follows_selection():
    S = np.zeros((2000, 36, 36))
    S[:, 3, 7] = 1.0
    v, o = draw_orientations(S, np.random.default_rng(1))
    assert np.all((v >= 30) & (v < 40) & (o >= 70) & (o < 80))


def test_prior_transform_bounds():
    prior = Prior()
    lo, hi = prior.transform(np.zeros(len(PARAMS))), prior.transform(np.ones(len(PARAMS)))
    for k, name in enumerate(PARAMS):
        assert min(lo[k], hi[k]) == pytest.approx(min(getattr(prior, name)))
        assert max(lo[k], hi[k]) == pytest.approx(max(getattr(prior, name)))


def test_nondetection_bright_planet_is_found_faint_one_missed():
    nd = NonDetection()
    # Neptune-mass planet at 250 AU on the ecliptic would be V ~ 17: found unless in a gap
    assert nd.p_missed(17.1, 250.0, 0.0, 0.0, 0.0, 0.0) < 0.5
    # an Earth-mass planet at 1000 AU is far below the limit
    assert nd.p_missed(1.0, 1000.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(1.0)
    assert planet_H(17.1, 0.3) < planet_H(5.0, 0.3)
