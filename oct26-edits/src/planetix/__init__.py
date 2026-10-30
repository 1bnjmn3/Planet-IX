"""planetix: tools for testing the Planet Nine hypothesis against real ETNO orbits.

Subpackages
-----------
data      Orbit catalogues (MPC parser, JPL SBDB client) and the canonical ETNO sample.
stats     Circular statistics, null tests, PCA/clustering.
dynamics  Secular theory and N-body checks.
bias      Survey-selection modelling.
"""

from planetix.angles import varpi_deg, wrap_deg

__version__ = "0.1.0"
__all__ = ["varpi_deg", "wrap_deg", "__version__"]
