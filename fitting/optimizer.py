"""
Optimization utilities for fitting log-Gamma approximation coefficients.

This module wraps Differential Evolution for coefficient optimization.

The optimizer is intentionally independent of:
    - approximation models
    - routing
    - Gamma identities
    - loss definitions

It receives a loss object and minimizes it over bounded coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scipy.optimize import differential_evolution

from utils.config import (
    DE_MAXITER,
    DE_POPSIZE,
    DE_POLISH,
    DE_SEED,
    DE_TOL,
)

from .losses import Loss


@dataclass(frozen=True)
class OptimizerResult:
    """
    Result returned by coefficient optimization.

    Parameters
    ----------
    coefficients
        Optimized coefficient vector.

    loss
        Final objective value.

    success
        Whether optimization succeeded.

    message
        Optimizer termination message.

    iterations
        Number of DE iterations performed.
    """

    coefficients: NDArray[np.float64]

    loss: float

    success: bool

    message: str

    iterations: int



def fit_coefficients(
    loss: Loss,
    objective_args: dict[str, Any],
    num_coefficients: int,
    initial_population: NDArray[np.float64] | None = None,
) -> OptimizerResult:
    """
    Optimize approximation coefficients.

    Parameters
    ----------
    loss
        Loss function to minimize.

    objective_args
        Arguments passed into the loss evaluation.

    num_coefficients
        Number of coefficients.

    initial_population
        Optional initial DE population.

        Providing the same population allows fair comparison
        between different loss functions.

    Returns
    -------
    OptimizerResult
    """


    bounds = [
        (0.0, 1.0)
        for _ in range(num_coefficients)
    ]


    def objective(
        coefficients: NDArray[np.float64],
    ) -> float:

        return loss.evaluate(
            coefficients=coefficients,
            **objective_args,
        )


    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=DE_SEED,
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        polish=DE_POLISH,
        mutation=(0.5, 0.9),
        recombination=0.9,
        init=(
            initial_population
            if initial_population is not None
            else "latinhypercube"
        ),
    )


    return OptimizerResult(
        coefficients=np.asarray(
            result.x,
            dtype=np.float64,
        ),

        loss=float(
            result.fun
        ),

        success=bool(
            result.success
        ),

        message=str(
            result.message
        ),

        iterations=int(
            result.nit
        ),
    )


__all__ = [
    "OptimizerResult",
    "fit_coefficients",
]