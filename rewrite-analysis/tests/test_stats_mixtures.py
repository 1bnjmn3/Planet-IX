import os

import numpy as np
import pytest

from planetix.stats import mixtures as M


def test_uniform_loglik():
    assert M.UNIFORM.loglik(np.zeros(5)) == pytest.approx(-5 * np.log(2 * np.pi))


def test_mixture_density_normalised():
    x = np.linspace(0, 360, 20001)[:-1]
    ll = np.array([M.LEGACY_BIMODAL_P9.loglik([v]) for v in x[::50]])
    assert np.exp(ll).mean() * 2 * np.pi == pytest.approx(1.0, rel=1e-3)


def test_em_recovers_two_components():
    truth = M.VonMisesMixture(weights=(0.5, 0.4), mus=(60.0, 240.0), kappas=(6.0, 6.0), w_uniform=0.1)
    x = truth.sample(600, rng=0)
    fit = M.fit_vonmises_mixture(x, 2, rng=0)
    assert sorted(round(m / 10) * 10 for m in fit.model.mus) == [60, 240]
    assert fit.n_params == 6
    assert M.fit_vonmises_mixture(x, 2, rng=0).bic < M.fit_vonmises_mixture(x, 1, rng=0).bic


@pytest.mark.skipif(not os.environ.get("PLANETIX_SLOW"), reason="slow calibration run; set PLANETIX_SLOW=1")
def test_lrt_calibrated_under_uniform():
    rng = np.random.default_rng(1)
    p = [M.mixture_lrt(rng.uniform(0, 360, 20), 0, 1, n_boot=39, n_init=3, rng=rng).p_value for _ in range(40)]
    assert np.mean(np.array(p) < 0.1) <= 0.25


def test_lrt_detects_cluster():
    x = M.VonMisesMixture((0.7,), (100.0,), (5.0,), 0.3).sample(30, rng=2)
    assert M.mixture_lrt(x, 0, 1, n_boot=49, n_init=3, rng=0).p_value < 0.05


def test_fixed_model_bayes_factor_sign():
    x_p9 = M.LEGACY_BIMODAL_P9.sample(30, rng=3)
    x_u = np.random.default_rng(3).uniform(0, 360, 30)
    assert M.compare_fixed_model(x_p9, M.LEGACY_BIMODAL_P9, n_sims=2000, rng=0).log10_bayes_factor > 1
    assert M.compare_fixed_model(x_u, M.LEGACY_BIMODAL_P9, n_sims=2000, rng=0).log10_bayes_factor < 0
