"""Bayesian model comparison: a distant planet against no planet, with survey selection.

kepler        two-body positions and magnitudes
selection     per-object selection functions S_j(varpi, Omega)
model         the likelihood ratio and priors
nondetection  probability that direct searches missed the planet
evidence      nested-sampling Bayes factors and simulated catalogues
"""

from planetix.inference.evidence import EvidenceResult, run_evidence
from planetix.inference.model import PARAMS, Likelihood, Prior, a_crit

__all__ = ["EvidenceResult", "Likelihood", "PARAMS", "Prior", "a_crit", "run_evidence"]
