"""Machine-learning tests of orbital clustering, calibrated against bias-aware nulls.

Replaces old script 02 (a TensorFlow autoencoder whose reconstruction error was read as
"Planet 9 interaction" without any calibration). Both tools here return a p-value:

``autoencoder_test``
    Autoencoder trained on mock samples from a ``BiasNull``; the real sample's mean
    reconstruction error is compared with that of fresh null mocks.
``classifier_test``
    Classifier two-sample test (Lopez-Paz & Oquab 2017): can a classifier tell the real
    orbits from null mocks better than it can tell null mocks from each other?
"""

from planetix.ml.autoencoder import autoencoder_test
from planetix.ml.c2st import classifier_test
from planetix.ml.features import orbit_features

__all__ = ["autoencoder_test", "classifier_test", "orbit_features"]
