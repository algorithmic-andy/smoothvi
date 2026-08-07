"""
Baseline functions for log-Gamma approximation.

This module defines the baseline components used by the approximation
models. A baseline captures the dominant behaviour of log(Gamma(x))
over the approximation interval, while a residual correction models the
remaining approximation error.

Each baseline provides its value together with its first and second
derivatives.

Available baselines
-------------------
- Polynomial Baseline
- Stirling Baseline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import LOG_SQRT_PI_OVER_TWO


@dataclass(frozen=True)
class Baseline:
    """
    Mathematical baseline function.

    Parameters
    ----------
    identifier
        Unique identifier used internally and for serialization.

    display_name
        Human-readable name used in plots and tables.

    value
        Function evaluating the baseline.

    first
        Function evaluating the first derivative.

    second
        Function evaluating the second derivative.
    """

    identifier: str
    display_name: str

    value: Callable[[ArrayLike], NDArray[np.float64]]
    first: Callable[[ArrayLike], NDArray[np.float64]]
    second: Callable[[ArrayLike], NDArray[np.float64]]

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the baseline."""
        return self.value(x)


# ---------------------------------------------------------------------
# Polynomial Baseline
# ---------------------------------------------------------------------

_POLYNOMIAL_CONSTANT = -4.0 * LOG_SQRT_PI_OVER_TWO
_POLYNOMIAL_SECOND = 2.0 * _POLYNOMIAL_CONSTANT


def _polynomial_value(x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return _POLYNOMIAL_CONSTANT * (x - 1.0) * (x - 2.0)


def _polynomial_first(x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return _POLYNOMIAL_CONSTANT * (2.0 * x - 3.0)


def _polynomial_second(x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return np.full_like(x, _POLYNOMIAL_SECOND)


# ---------------------------------------------------------------------
# Stirling Baseline
# ---------------------------------------------------------------------


def _stirling_value(x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return (x - 0.5) * np.log(x) - x + 1.0


def _stirling_first(x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return np.log(x) - 0.5 / x


def _stirling_second(x: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / x + 0.5 / (x * x)


# ---------------------------------------------------------------------
# Public baseline objects
# ---------------------------------------------------------------------

POLYNOMIAL_BASELINE = Baseline(
    identifier="polynomial",
    display_name="Polynomial Baseline",
    value=_polynomial_value,
    first=_polynomial_first,
    second=_polynomial_second,
)

STIRLING_BASELINE = Baseline(
    identifier="stirling",
    display_name="Stirling Baseline",
    value=_stirling_value,
    first=_stirling_first,
    second=_stirling_second,
)

__all__ = [
    "Baseline",
    "POLYNOMIAL_BASELINE",
    "STIRLING_BASELINE",
]