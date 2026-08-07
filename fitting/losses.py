"""
Loss functions for fitting log-Gamma approximations.

Three fitting objectives are implemented:

1. Least-squares fitting against log(Gamma).
2. Log-Legendre duplication identity fitting.
3. Gamma boundary, Digamma boundary, Trigamma boundary, log-Legendre duplication identity fitting.

All losses operate over the evaluation interval [1,2].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from scipy.special import gammaln

from approximation.precompute import ApproximationCache

from utils.config import IDENTITY_WEIGHTS, LEAST_SQUARE_WEIGHTS


@dataclass(frozen=True)
class Loss:
    """
    Mathematical fitting objective.
    """

    identifier: str
    display_name: str

    evaluate: Callable[..., float]



def _evaluate_cache(
    cache: ApproximationCache,
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate an approximation from cached quantities.
    """

    value = (
        cache.baseline_value
        +
        cache.residual_basis @ coefficients
    )

    if cache.sign is not None:

        value = (
            cache.sign * value
        )

    if cache.correction is not None:

        value = (
            value
            +
            cache.correction
        )

    return value


def _evaluate_cache_first(
    cache: ApproximationCache,
    coefficients: NDArray[np.float64],
    ) -> NDArray[np.float64]:
    """
    Evaluate the cached first derivative approximation.
    """

    return (
        cache.baseline_first
        + cache.residual_first @ coefficients
    )


def _evaluate_cache_second(
        cache: ApproximationCache,
        coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate the cached second derivative approximation.
    """

    return (
        cache.baseline_second
        + cache.residual_second @ coefficients
    )


def _least_squares_loss(
    cache: ApproximationCache,
    coefficients: NDArray[np.float64],
    target: NDArray[np.float64],
) -> float:
    """
    Least squares log-Gamma fitting loss.
    """

    approximation = _evaluate_cache(
        cache,
        coefficients,
    )
    approximation_first = _evaluate_cache_first(
        cache,
        coefficients,
    )
    approximation_second = _evaluate_cache_second(
        cache,
        coefficients,
    )

    target_loggam = target[0]
    target_digam = target[1]
    target_trigam = target[2]

    loggam_result = (approximation - target_loggam) ** 2
    digam_result = (approximation_first - target_digam) ** 2
    trigam_result = (approximation_second - target_trigam) ** 2

    norm_factor = sum(LEAST_SQUARE_WEIGHTS)

    loggam_weight = LEAST_SQUARE_WEIGHTS[0]
    digam_weight = LEAST_SQUARE_WEIGHTS[1]
    trigam_weight = LEAST_SQUARE_WEIGHTS[2]

    least_square_loss = loggam_result * loggam_weight
    least_square_loss += digam_result * digam_weight
    least_square_loss += trigam_result * trigam_weight
    least_square_loss /= norm_factor

    return least_square_loss



def _duplication_loss(
    cache_x: ApproximationCache,
    cache_half: ApproximationCache,
    cache_two: ApproximationCache,
    coefficients: NDArray[np.float64],
) -> float:
    """
    Log-Legendre duplication identity loss.
    """

    gx = _evaluate_cache(
        cache_x,
        coefficients,
    )

    gx_half = _evaluate_cache(
        cache_half,
        coefficients,
    )

    g_two = _evaluate_cache(
        cache_two,
        coefficients,
    )

    x = cache_x.x

    identity_constant = (
        0.5 * np.log(np.pi)
        +
        (1.0 - 2.0 * x) * np.log(2.0)
    )

    violation = (
        gx
        +
        gx_half
        -
        g_two
        -
        identity_constant
    )

    return float(
        np.mean(
            violation ** 2
        )
    )

def _gamma_boundary_loss(
        cache_x: ApproximationCache,
        coefficients: NDArray[np.float64],
) -> float:
    """
    Gamma boundary violation loss.
    """
        
    approximation = _evaluate_cache(cache_x, coefficients)

    lower_violation = np.maximum(cache_x.gamma_lower - approximation, 0.0)
    upper_violation = np.maximum(approximation - cache_x.gamma_upper, 0.0)

    return float(
        np.mean(
            lower_violation**2 + upper_violation**2
        )
    )


def _digamma_boundary_loss(
       cache_x: ApproximationCache,
       coefficients: NDArray[np.float64],
) -> float:
    """
    Digamma boundary violation loss.
    """
        
    approximation = _evaluate_cache_first(cache_x, coefficients)

    lower_violation = np.maximum(cache_x.digamma_lower - approximation, 0.0)
    upper_violation = np.maximum(approximation - cache_x.digamma_upper, 0.0)

    return float(
        np.mean(
            lower_violation**2 + upper_violation**2
        )
    )


def _trigamma_boundary_loss(
        cache_x: ApproximationCache,
        coefficients: NDArray[np.float64],
) -> float:
    """
    Trigamma boundary violation loss.
    """

    approximation = _evaluate_cache_second(cache_x, coefficients)

    lower_violation = np.maximum(cache_x.trigamma_lower - approximation, 0.0)
    upper_violation = np.maximum(approximation - cache_x.trigamma_upper, 0.0)

    return float(
        np.mean(
            lower_violation**2 + upper_violation**2
        )
    )


def _identity_constrained_loss(
    cache_x: ApproximationCache,
    cache_half: ApproximationCache,
    cache_two: ApproximationCache,
    coefficients: NDArray[np.float64],
    ) -> float:
    """
    Total identity constrained loss.
    """

    norm_factor = sum(IDENTITY_WEIGHTS)

    dup_loss = _duplication_loss(cache_x, cache_half, cache_two, coefficients)
    dup_weight = IDENTITY_WEIGHTS[0]

    gam_bound_loss = _gamma_boundary_loss(cache_x, coefficients)
    gam_bound_weight = IDENTITY_WEIGHTS[1]

    digam_bound_loss = _digamma_boundary_loss(cache_x, coefficients)
    digam_bound_weight = IDENTITY_WEIGHTS[2]

    trigam_bound_loss = _trigamma_boundary_loss(cache_x, coefficients)
    trigam_bound_weight = IDENTITY_WEIGHTS[3]

    identity_loss = dup_loss * dup_weight
    identity_loss += gam_bound_loss * gam_bound_weight
    identity_loss += digam_bound_loss * digam_bound_weight
    identity_loss += trigam_bound_loss * trigam_bound_weight
    identity_loss /= norm_factor

    return identity_loss



LEAST_SQUARES_LOSS = Loss(
    identifier="least_squares",
    display_name="Least Squares",
    evaluate=_least_squares_loss,
)


DUPLICATION_LOSS = Loss(
    identifier="duplication",
    display_name="Log-Legendre Duplication Identity",
    evaluate=_duplication_loss,
)


IDENTITY_CONSTRAINED_LOSS = Loss(
    identifier="identity_constrained",
    display_name="Identity Constrained",
    evaluate=_identity_constrained_loss,
)


ALL_LOSSES = [
    LEAST_SQUARES_LOSS,
    DUPLICATION_LOSS,
    IDENTITY_CONSTRAINED_LOSS,
]


__all__ = [
    "Loss",
    "LEAST_SQUARES_LOSS",
    "DUPLICATION_LOSS",
    "IDENTITY_CONSTRAINED_LOSS",
    "ALL_LOSSES",
]