"""Survey-selection modelling and bias-aware null tests for orbital clustering.

Two independent selection models:

``EmpiricalFootprint``
    Brown (2017)-style footprint built from the discovery positions and magnitudes of
    every known TNO in the MPC snapshot. Covers any object.
``SurveySet``
    The characterised OSSOS, CFEPS, High-Latitude and MA surveys (OSSOS v11 release):
    real pointings, epochs and efficiency functions. Valid only for objects those
    surveys found.

Both plug into ``selection_pool`` / ``selection_grid`` (per-object selection functions)
and ``BiasNull`` / ``clustering_test`` (p-values against a bias-aware null).
"""

from planetix.bias.footprint import EmpiricalFootprint, designation_date_jd, discovery_table
from planetix.bias.inclination import fit_hot_disk, inclination_selection
from planetix.bias.nulls import BiasNull, clustering_statistics, clustering_test, mean_vectors
from planetix.bias.selection import clone_weights, selection_grid, selection_pool
from planetix.bias.surveys import SurveySet

__all__ = [
    "EmpiricalFootprint", "designation_date_jd", "discovery_table", "fit_hot_disk",
    "inclination_selection", "BiasNull", "clustering_statistics", "clustering_test", "mean_vectors",
    "clone_weights", "selection_grid", "selection_pool", "SurveySet",
]
