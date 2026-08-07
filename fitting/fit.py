"""
Experiment driver for fitting log-Gamma approximation models.

This module produces fitted coefficient sets for every combination of:

    baseline
    residual family
    fitting objective

using identical Differential Evolution initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scipy.special import gammaln, digamma, polygamma

from approximation.model import (
    LogGammaApproximation,
    ALL_APPROXIMATIONS,
)

from approximation.precompute import (
    build_cache,
)

from utils.routing import (
    route_to_core_interval,
)

from approximation.constants import (
    NUM_COEFFICIENTS,
    APPROXIMATION_INTERVAL,
)

from utils.config import (
    PRECOMPUTE_GRID_SIZE,
)

from .losses import (
    Loss,
    LEAST_SQUARES_LOSS,
    DUPLICATION_LOSS,
    IDENTITY_CONSTRAINED_LOSS,
)

from .optimizer import (
    OptimizerResult,
    fit_coefficients,
)

from .population import (
    generate_initial_population,
)



@dataclass(frozen=True)
class FitResult:
    """
    Stores one completed approximation fit.
    """

    model_id: str

    loss_id: str

    coefficients: NDArray[np.float64]

    objective_value: float

    success: bool

    message: str



def _build_grid() -> NDArray[np.float64]:
    """
    Construct evaluation grid over [1,2].
    """

    return np.linspace(
        APPROXIMATION_INTERVAL[0],
        APPROXIMATION_INTERVAL[1],
        PRECOMPUTE_GRID_SIZE,
    )



def _build_least_squares_args(
    model: LogGammaApproximation,
    x: NDArray[np.float64],
) -> dict[str, Any]:
    """
    Build least squares cache.
    """

    cache = build_cache(
        model,
        x,
    )

    target = [
        gammaln(x),
        digamma(x),
        polygamma(1, x),
    ]

    return {
        "cache": cache,
        "target": target,
    }



def _build_duplication_args(
    model: LogGammaApproximation,
    x: NDArray[np.float64],
) -> dict[str, Any]:
    """
    Build duplication identity caches.
    """

    x_half = x + 0.5

    x_two = 2.0 * x


    routed_x = route_to_core_interval(
        x
    )

    routed_half = route_to_core_interval(
        x_half
    )

    routed_two = route_to_core_interval(
        x_two
    )


    cache_x = build_cache(
        model,
        routed_x.x,
        routed_x.correction,
        routed_x.sign,
    )

    cache_half = build_cache(
        model,
        routed_half.x,
        routed_half.correction,
        routed_half.sign,
    )

    cache_two = build_cache(
        model,
        routed_two.x,
        routed_two.correction,
        routed_two.sign,
    )


    return {
        "cache_x": cache_x,
        "cache_half": cache_half,
        "cache_two": cache_two,
    }



def fit_all_models() -> list[FitResult]:
    """
    Fit every approximation model under every loss.

    Returns
    -------
    results
        List containing all fitted coefficient sets.
    """

    x = _build_grid()
    print(x)


    initial_population = generate_initial_population(
        NUM_COEFFICIENTS,
    )


    results = []


    for model in ALL_APPROXIMATIONS:

        for loss in [
            LEAST_SQUARES_LOSS,
            DUPLICATION_LOSS,
            IDENTITY_CONSTRAINED_LOSS,
        ]:


            if loss.identifier == "least_squares":

                objective_args = (
                    _build_least_squares_args(
                        model,
                        x,
                    )
                )

            else:

                objective_args = (
                    _build_duplication_args(
                        model,
                        x,
                    )
                )


            result: OptimizerResult = fit_coefficients(
                loss,
                objective_args,
                NUM_COEFFICIENTS,
                initial_population,
            )


            results.append(
                FitResult(
                    model_id=model.identifier,
                    loss_id=loss.identifier,
                    coefficients=result.coefficients,
                    objective_value=result.loss,
                    success=result.success,
                    message=result.message,
                )
            )


    return results



__all__ = [
    "FitResult",
    "fit_all_models",
]