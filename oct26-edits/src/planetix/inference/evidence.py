"""Bayesian evidence for the planet model, and simulated catalogues for calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetix.inference.model import PARAMS, Likelihood, Prior


@dataclass
class EvidenceResult:
    log_bf: float  # ln Z1 - ln Z0
    log_bf_err: float
    samples: np.ndarray  # equal-weight posterior samples, columns as PARAMS
    n_obj: int

    @property
    def bf(self) -> float:
        return float(np.exp(self.log_bf))

    def p_planet(self, prior_odds: float = 1.0, extra_factor: float = 1.0) -> float:
        """Posterior probability of H1 for given prior odds P(H1)/P(H0)."""
        odds = prior_odds * self.bf * extra_factor
        return odds / (1.0 + odds)


def run_evidence(like: Likelihood, prior: Prior | None = None, nlive: int = 500, dlogz: float = 0.05,
                 seed: int = 0, bound: str = "multi", sample: str = "rwalk") -> EvidenceResult:
    """Nested sampling (dynesty) of the H1 evidence relative to H0.

    The likelihood is already the ratio p(D | theta, H1) / p(D | H0), so the
    evidence it returns is the Bayes factor Z1 / Z0 directly.
    """
    import dynesty

    prior = prior or Prior()
    sampler = dynesty.NestedSampler(like, prior.transform, len(PARAMS), nlive=nlive, bound=bound,
                                    sample=sample, rstate=np.random.default_rng(seed))
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    res = sampler.results
    samples = res.samples_equal(rstate=np.random.default_rng(seed + 1))
    return EvidenceResult(float(res.logz[-1]), float(res.logzerr[-1]), samples, len(like.a))


def draw_orientations(S: np.ndarray, rng, g=None) -> tuple[np.ndarray, np.ndarray]:
    """Draw one (varpi, Omega) per object from S_j (times g_j if given), in degrees.

    With g = None this simulates a catalogue from the no-planet hypothesis seen
    through the same selection function as the real data.
    """
    n, nv, no = S.shape
    P = S if g is None else S * g
    flat = P.reshape(n, -1)
    flat = flat / flat.sum(axis=1, keepdims=True)
    cdf = np.cumsum(flat, axis=1)
    k = (cdf < rng.random((n, 1))).sum(axis=1).clip(max=nv * no - 1)
    iv, io = np.divmod(k, no)
    varpi = (iv + rng.random(n)) * 360.0 / nv
    Omega = (io + rng.random(n)) * 360.0 / no
    return varpi, Omega


def planet_density_grid(like: Likelihood, theta) -> np.ndarray:
    """g_j(varpi, Omega) on the selection grid for parameters theta, shape like S."""
    from planetix.inference.model import _log_vm_rel

    p = dict(zip(PARAMS, theta))
    F = like.confined_fraction(theta)
    gv, go = (np.radians((np.arange(n) + 0.5) * 360.0 / n) for n in like.S.shape[1:])
    hv = np.exp(_log_vm_rel(gv - np.radians(p["varpi9"] + 180), 10 ** p["log10_kvarpi"]))
    ko = 10 ** p["log10_kpole"] * np.sin(np.radians(like.i)) * np.sin(np.radians(p["i9"]))
    ho = np.exp(_log_vm_rel(go[None, :] - np.radians(p["Omega9"]), ko[:, None]))
    return (1 - F)[:, None, None] + F[:, None, None] * hv[None, :, None] * ho[:, None, :]
