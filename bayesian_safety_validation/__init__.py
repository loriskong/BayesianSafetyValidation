"""A Python implementation of bayesian safety validation."""
from .bayesian_safety_validation import BayesianSafetyValidation

import importlib.metadata
__version__ = importlib.metadata.version('bayesian-safety-validation')


__all__ = [
    "BayesianSafetyValidation",
]