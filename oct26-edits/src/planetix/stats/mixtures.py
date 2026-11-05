"""Von Mises mixture models for clustered angles.

These replace the random-forest "Planet 9 probability" of old scripts 09, 12
and 14. That number was the average classifier output on real objects after
training on a 50/50 mix of two simulated worlds; it depends on the arbitrary
50/50 prior and on how the worlds were simulated, and it is not a probability
that a planet exists. The questions it was after have direct likelihood
answers:

* *Does a specific prescribed model (e.g. 60% at 60 deg, 30% at 240 deg, 10%
  scatter) explain the angles better than uniform?*  With no free parameters
  the likelihood ratio **is** the Bayes factor: ``compare_fixed_model``.
* *Is there one cluster? two?*  Fit mixtures with k = 0, 1, 2 components plus
  an optional uniform background (``fit_vonmises_mixture``), and calibrate the
  likelihood ratio with a parametric bootstrap (``mixture_lrt``). Wilks'
  theorem does not apply because the null sits on a parameter boundary
  (kappa = 0 or weight = 0), so chi-square p-values would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from planetix.angles import wrap_deg
from planetix.stats.circular import _a1inv
from planetix.stats.nulls import mc_p_value

_KAPPA_MAX = 500.0
_LOG2PI = np.log(2 * np.pi)


def _vm_logpdf(th, mu, kappa):
    return kappa * (np.cos(th - mu) - 1) - np.log(2 * np.pi * special.i0e(kappa))


@dataclass(frozen=True)
class VonMisesMixture:
    """Mixture of k von Mises components plus an optional uniform background.

    ``weights`` has k entries for the components; ``w_uniform`` is the
    background weight (0 when there is no background). Angles in degrees.
    """

    weights: tuple[float, ...]
    mus: tuple[float, ...]
    kappas: tuple[float, ...]
    w_uniform: float = 0.0

    @property
    def k(self) -> int:
        return len(self.mus)

    def n_params(self, background: bool) -> int:
        # mu and kappa per component, plus free weights
        return 2 * self.k + (self.k if background else max(self.k - 1, 0))

    def _component_logpdf(self, th):
        cols = [np.log(w) + _vm_logpdf(th, np.deg2rad(m), k) for w, m, k in zip(self.weights, self.mus, self.kappas) if w > 0]
        if self.w_uniform > 0:
            cols.append(np.full_like(th, np.log(self.w_uniform) - _LOG2PI))
        return np.column_stack(cols)

    def loglik(self, theta_deg) -> float:
        th = np.deg2rad(np.asarray(theta_deg, dtype=float).ravel())
        if self.k == 0:
            return float(-th.size * _LOG2PI)
        return float(special.logsumexp(self._component_logpdf(th), axis=1).sum())

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        if self.k == 0:
            return rng.uniform(0, 360, n)
        p = np.r_[self.weights, self.w_uniform]
        which = rng.choice(p.size, size=n, p=p / p.sum())
        out = rng.uniform(0, 2 * np.pi, n)
        for j in range(self.k):
            sel = which == j
            out[sel] = rng.vonmises(np.deg2rad(self.mus[j]), self.kappas[j], sel.sum())
        return wrap_deg(np.rad2deg(out))


UNIFORM = VonMisesMixture(weights=(), mus=(), kappas=(), w_uniform=1.0)


@dataclass(frozen=True)
class MixtureFit:
    model: VonMisesMixture
    loglik: float
    n_params: int
    bic: float
    aic: float
    responsibilities: np.ndarray


def fit_vonmises_mixture(theta_deg, k: int, background: bool = True, n_init: int = 20, max_iter: int = 200, rng=None) -> MixtureFit:
    """Maximum-likelihood fit by EM with random restarts.

    ``k = 0`` returns the uniform model. With ``background`` a uniform
    component is always included, which keeps a single stray object from
    being forced into a cluster. Concentrations are capped at 500 to avoid
    the degenerate infinite-likelihood solution where one component collapses
    onto a single point.
    """
    th = np.deg2rad(np.asarray(theta_deg, dtype=float).ravel())
    n = th.size
    rng = np.random.default_rng(rng)

    def finish(model, ll, resp):
        p = model.n_params(background)
        return MixtureFit(model, ll, p, -2 * ll + p * np.log(n), -2 * ll + 2 * p, resp)

    if k == 0:
        return finish(UNIFORM, UNIFORM.loglik(np.rad2deg(th)), np.ones((n, 1)))

    best = None
    for _ in range(n_init):
        mus = rng.choice(th, size=k, replace=False) if n >= k else rng.uniform(0, 2 * np.pi, k)
        kap = np.full(k, 2.0)
        w = np.full(k + background, 1.0 / (k + background))
        ll_old = -np.inf
        for _ in range(max_iter):
            logp = [np.log(w[j]) + _vm_logpdf(th, mus[j], kap[j]) for j in range(k)]
            if background:
                logp.append(np.full(n, np.log(w[k]) - _LOG2PI))
            logp = np.column_stack(logp)
            norm = special.logsumexp(logp, axis=1)
            ll = norm.sum()
            resp = np.exp(logp - norm[:, None])
            w = np.clip(resp.mean(axis=0), 1e-12, None)
            w /= w.sum()
            for j in range(k):
                r = resp[:, j]
                C, S = np.sum(r * np.cos(th)), np.sum(r * np.sin(th))
                mus[j] = np.arctan2(S, C)
                rbar = np.hypot(C, S) / max(r.sum(), 1e-12)
                kap[j] = min(_a1inv(min(rbar, 1 - 1e-9)), _KAPPA_MAX)
            if ll - ll_old < 1e-6:
                break
            ll_old = ll
        if best is None or ll > best[0]:
            model = VonMisesMixture(
                weights=tuple(float(x) for x in w[:k]),
                mus=tuple(float(x) for x in wrap_deg(np.rad2deg(mus))),
                kappas=tuple(float(x) for x in kap),
                w_uniform=float(w[k]) if background else 0.0,
            )
            best = (ll, model, resp)
    # re-evaluate with final parameters so loglik matches the reported model
    model = best[1]
    return finish(model, model.loglik(np.rad2deg(th)), best[2])


@dataclass(frozen=True)
class MixtureLRT:
    k0: int
    k1: int
    lr_stat: float
    p_value: float
    null: np.ndarray
    fit0: MixtureFit
    fit1: MixtureFit


def mixture_lrt(theta_deg, k0: int, k1: int, background: bool = True, n_boot: int = 500, n_init: int = 10, rng=None) -> MixtureLRT:
    """Parametric-bootstrap likelihood ratio test of k0 vs k1 components.

    Fits both models, simulates ``n_boot`` datasets of the same size from the
    fitted k0 model, refits both on each, and compares 2 (ll1 - ll0).
    """
    if k1 <= k0:
        raise ValueError("k1 must exceed k0")
    th = np.asarray(theta_deg, dtype=float).ravel()
    rng = np.random.default_rng(rng)
    f0 = fit_vonmises_mixture(th, k0, background, n_init, rng=rng)
    f1 = fit_vonmises_mixture(th, k1, background, n_init, rng=rng)
    obs = max(2 * (f1.loglik - f0.loglik), 0.0)
    null = np.empty(n_boot)
    for b in range(n_boot):
        x = f0.model.sample(th.size, rng)
        l0 = fit_vonmises_mixture(x, k0, background, n_init, rng=rng).loglik
        l1 = fit_vonmises_mixture(x, k1, background, n_init, rng=rng).loglik
        null[b] = max(2 * (l1 - l0), 0.0)
    return MixtureLRT(k0, k1, float(obs), mc_p_value(obs, null), null, f0, f1)


@dataclass(frozen=True)
class FixedModelComparison:
    log10_bayes_factor: float
    p_value_vs_uniform: float
    null_log10_bf: np.ndarray


def compare_fixed_model(theta_deg, model: VonMisesMixture, n_sims: int = 20000, rng=None) -> FixedModelComparison:
    """Bayes factor of a fully specified model against the uniform null.

    Positive log10 BF favours ``model``; Jeffreys' scale calls |log10 BF| < 0.5
    barely worth mentioning and > 1 strong. ``p_value_vs_uniform`` is the
    frequentist calibration: how often uniform data of the same size give a
    BF at least this large.
    """
    th = np.asarray(theta_deg, dtype=float).ravel()
    rng = np.random.default_rng(rng)
    n = th.size

    def log10bf(x):
        return (model.loglik(x) + n * _LOG2PI) / np.log(10)

    obs = log10bf(th)
    null = np.array([log10bf(rng.uniform(0, 360, n)) for _ in range(n_sims)])
    return FixedModelComparison(float(obs), mc_p_value(obs, null), null)


# The prescribed "bimodal Planet 9" world of old scripts 12 and 14.
LEGACY_BIMODAL_P9 = VonMisesMixture(weights=(0.6, 0.3), mus=(60.0, 240.0), kappas=(2.5, 2.0), w_uniform=0.1)
# The unimodal world of old script 09.
LEGACY_UNIMODAL_P9 = VonMisesMixture(weights=(1.0,), mus=(60.0,), kappas=(2.0,), w_uniform=0.0)
