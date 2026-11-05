import numpy as np
import pandas as pd

from planetix.bias.inclination import fit_hot_disk
from planetix.bias.nulls import BiasNull, clustering_test, mean_vectors


def _sample(Omega, omega, i=15.0):
    n = len(Omega)
    return pd.DataFrame({"designation": [f"o{k}" for k in range(n)], "a": 400.0, "e": 0.9, "i": np.full(n, i),
                         "Omega": Omega, "omega": omega})


def test_uniform_null_gives_uniform_p_values():
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(60):
        s = _sample(rng.uniform(0, 360, 20), rng.uniform(0, 360, 20))
        ps.append(clustering_test(s, BiasNull.uniform(s, rng=rng), 500, rng).loc["R_varpi", "p"])
    ps = np.array(ps)
    assert 0.3 < ps.mean() < 0.7 and (ps < 0.05).mean() < 0.15


def test_planted_cluster_is_detected():
    rng = np.random.default_rng(2)
    s = _sample(rng.normal(100, 20, 20) % 360, rng.normal(300, 20, 20) % 360)
    res = clustering_test(s, BiasNull.uniform(s, rng=rng), 2000, rng)
    assert res.loc["R_varpi", "p"] < 0.01 and res.loc["min_p", "p"] < 0.01


def test_bias_explains_a_cluster_it_created():
    """If the null's pools are themselves clustered, a sample drawn from them is not significant on D."""
    rng = np.random.default_rng(3)
    n = 20
    s = _sample(np.zeros(n), np.zeros(n))
    pools_O = rng.normal(100, 30, (n, 3000)) % 360
    pools_w = rng.normal(300, 30, (n, 3000)) % 360
    null = BiasNull(s["designation"].to_numpy(), s["i"].to_numpy(), pools_O, pools_w)
    Om, w = null.draw(1, rng)
    s["Omega"], s["omega"] = Om[0], w[0]
    biased = clustering_test(s, null, 2000, rng)
    uniform = clustering_test(s, BiasNull.uniform(s, rng=rng), 2000, rng)
    assert uniform.loc["R_varpi", "p"] < 0.01
    assert biased.loc["D_varpi", "p"] > 0.01


def test_sampler_shape():
    s = _sample(np.zeros(5), np.zeros(5))
    draw = BiasNull.uniform(s, rng=0).sampler("varpi")
    assert draw(np.random.default_rng(0)).shape == (5,)


def test_mean_vectors_shapes():
    v = mean_vectors(np.full((7, 4), 10.0), np.zeros((7, 4)), np.zeros((7, 4)))
    assert v["varpi"].shape == (7, 2) and v["pole"].shape == (7, 3)


def test_hot_disk_fit_recovers_width():
    rng = np.random.default_rng(0)
    x, u = rng.uniform(0, 90, 200000), rng.uniform(0, 1, 200000)
    i = x[u < np.sin(np.radians(x)) * np.exp(-0.5 * (x / 15.0) ** 2)][:400]  # rejection sampling
    fit = fit_hot_disk(i)
    assert fit["sigma_lo"] <= 15.0 <= fit["sigma_hi"]
