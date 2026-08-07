"""
Precomputation utilities for log-Gamma approximations.

This module caches all coefficient-independent evaluations of an
approximation model.

For a model

    f(x,c) = B(x) + R(x)c

we cache

    B(x)
    B'(x)
    B''(x)

    R(x)
    R'(x)
    R''(x)

Routing is handled separately. If an evaluation point has been
transported into the approximation interval, the corresponding
correction metadata can be stored alongside the cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import LogGammaApproximation


@dataclass(frozen=True)
class ApproximationCache:
    """
    Cached approximation evaluations.

    Parameters
    ----------
    x
        Evaluation locations. These should already lie inside the
        approximation interval.

    baseline_value
        Baseline values.

    baseline_first
        Baseline first derivatives.

    baseline_second
        Baseline second derivatives.

    residual_basis
        Residual basis matrix.

    residual_first
        First derivative residual basis matrix.

    residual_second
        Second derivative residual basis matrix.

    gamma_lower
        Gamma lower bound matrix.

    gamma_upper
        Gamma upper bound matrix.

    digamma_lower
        Digamma lower bound matrix.

    digamma_upper
        Digamma upper bound matrix.

    trigamma_lower
        Trigamma lower bound matrix.

    trigamma_upper
        Trigamma upper bound matrix.

    correction
        Optional additive routing correction.

    sign
        Optional routing sign metadata.
    """

    x: NDArray[np.float64]

    baseline_value: NDArray[np.float64]
    baseline_first: NDArray[np.float64]
    baseline_second: NDArray[np.float64]

    residual_basis: NDArray[np.float64]
    residual_first: NDArray[np.float64]
    residual_second: NDArray[np.float64]

    gamma_lower: NDArray[np.float64] | None = None
    gamma_upper: NDArray[np.float64] | None = None

    digamma_lower: NDArray[np.float64] | None = None
    digamma_upper: NDArray[np.float64] | None = None

    trigamma_lower: NDArray[np.float64] | None = None
    trigamma_upper: NDArray[np.float64] | None = None

    correction: NDArray[np.float64] | None = None
    sign: NDArray[np.float64] | None = None



def build_cache(
    model: LogGammaApproximation,
    x: ArrayLike,
    correction: NDArray[np.float64] | None = None,
    sign: NDArray[np.float64] | None = None,
) -> ApproximationCache:
    """
    Precompute all coefficient-independent quantities.

    Parameters
    ----------
    model
        Approximation model.

    x
        Evaluation locations.

    correction
        Optional routing correction.

    sign
        Optional routing sign metadata.

    Returns
    -------
    ApproximationCache
    """

    x = np.ascontiguousarray(
        x,
        dtype=np.float64,
    )

    gamma_lower = (
    0.5 * np.log(2.0 * np.pi)
    + (x - 0.5) * np.log(x)
    - x
    + 1.0 / (12.0 * x + 1.0)
    )

    gamma_upper = (
        0.5 * np.log(2.0 * np.pi)
        + (x - 0.5) * np.log(x)
        - x
        + 1.0 / (12.0 * x)
    )

    digamma_lower = (
        np.log(x)
        - 1.0 / (2.0 * x)
        - 1.0 / (12.0 * x**2)
        + 1.0 / (120.0 * x**4)
        - 1.0 / (252.0 * x**6)
    )

    digamma_upper = (
        np.log(x)
        - 1.0 / (2.0 * x)
        - 1.0 / (12.0 * x**2)
        + 1.0 / (120.0 * x**4)
    )

    trigamma_lower = (
        1.0 / x
        + 1.0 / (2.0 * x**2)
    )

    trigamma_upper = (
        1.0 / x
        + 1.0 / (x**2)
    )


    return ApproximationCache(

        x=x,

        baseline_value=model.baseline.value(x),
        baseline_first=model.baseline.first(x),
        baseline_second=model.baseline.second(x),

        residual_basis=model.residual.basis(x),
        residual_first=model.residual.first(x),
        residual_second=model.residual.second(x),

        gamma_lower = gamma_lower,
        gamma_upper = gamma_upper,

        digamma_lower = digamma_lower,
        digamma_upper = digamma_upper,

        trigamma_lower = trigamma_lower,
        trigamma_upper = trigamma_upper,

        correction=correction,
        sign=sign,
    )


__all__ = [
    "ApproximationCache",
    "build_cache",
]