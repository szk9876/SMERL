"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from .gaussian_mixture import GaussianMixture
from .bayesian_mixture import BayesianGaussianMixture
from .k_means_ import KMeans

__all__ = ['GaussianMixture',
           'BayesianGaussianMixture',
           'KMeans']
