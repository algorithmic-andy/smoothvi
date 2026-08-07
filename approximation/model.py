"""
Combined log-Gamma approximation models.

An approximation model combines a baseline function with a residual
correction:

    logGamma(x) ≈ Baseline(x) + Residual(x) @ coefficients

The model does not store coefficients. Coefficients are produced by
the fitting package and supplied during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import (
    Baseline,
    POLYNOMIAL_BASELINE,
    STIRLING_BASELINE,
)

from .residuals import (
    Residual,
    BERNSTEIN_RESIDUAL,
    RATIONAL_RESIDUAL,
)


@dataclass(frozen=True)
class LogGammaApproximation:
    """
    Combined baseline + residual approximation.

    Parameters
    ----------
    identifier
        Unique identifier for serialization.

    display_name
        Human-readable experiment label.

    baseline
        Baseline component.

    residual
        Residual correction component.
    """

    identifier: str
    display_name: str

    baseline: Baseline
    residual: Residual


    def __call__(
        self,
        x: ArrayLike,
        coefficients: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate approximation.
        """

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        coefficients = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        return (
            self.baseline(x)
            + self.residual.basis(x)
            @ coefficients
        )


    def first(
        self,
        x: ArrayLike,
        coefficients: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate first derivative.
        """

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        coefficients = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        return (
            self.baseline.first(x)
            + self.residual.first(x)
            @ coefficients
        )


    def second(
        self,
        x: ArrayLike,
        coefficients: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Evaluate second derivative.
        """

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        coefficients = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        return (
            self.baseline.second(x)
            + self.residual.second(x)
            @ coefficients
        )


# =====================================================================
# Public approximation models
# =====================================================================


POLYNOMIAL_BERNSTEIN = LogGammaApproximation(
    identifier="polynomial_bernstein",
    display_name="Polynomial Baseline + Bernstein Residual",
    baseline=POLYNOMIAL_BASELINE,
    residual=BERNSTEIN_RESIDUAL,
)


POLYNOMIAL_RATIONAL = LogGammaApproximation(
    identifier="polynomial_rational",
    display_name="Polynomial Baseline + Rational Residual",
    baseline=POLYNOMIAL_BASELINE,
    residual=RATIONAL_RESIDUAL,
)


STIRLING_BERNSTEIN = LogGammaApproximation(
    identifier="stirling_bernstein",
    display_name="Stirling Baseline + Bernstein Residual",
    baseline=STIRLING_BASELINE,
    residual=BERNSTEIN_RESIDUAL,
)


STIRLING_RATIONAL = LogGammaApproximation(
    identifier="stirling_rational",
    display_name="Stirling Baseline + Rational Residual",
    baseline=STIRLING_BASELINE,
    residual=RATIONAL_RESIDUAL,
)


ALL_APPROXIMATIONS = [
    POLYNOMIAL_BERNSTEIN,
    POLYNOMIAL_RATIONAL,
    STIRLING_BERNSTEIN,
    STIRLING_RATIONAL,
]


__all__ = [
    "LogGammaApproximation",
    "POLYNOMIAL_BERNSTEIN",
    "POLYNOMIAL_RATIONAL",
    "STIRLING_BERNSTEIN",
    "STIRLING_RATIONAL",
    "ALL_APPROXIMATIONS",
]