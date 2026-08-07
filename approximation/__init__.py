"""
Approximation package for structure-preserving log-Gamma approximations.

This package contains the mathematical components used to construct
approximation models for log(Gamma(x)), including:

- Baseline functions
- Residual correction functions
- Complete approximation models
- Precomputed evaluation utilities

The package is designed to support both:
1. Approximation coefficient fitting.
2. Downstream variational inference experiments.
"""

from .baselines import (
    Baseline,
    POLYNOMIAL_BASELINE,
    STIRLING_BASELINE,
)

__all__ = [
    "Baseline",
    "POLYNOMIAL_BASELINE",
    "STIRLING_BASELINE",
]