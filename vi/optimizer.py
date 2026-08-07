"""
Reference optimizer.

Computes λ* using the exact SciPy method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.optimize import minimize

from .elbo import (
    dirichlet_elbo,
)

from utils.config import (
    LAMBDA_FLOOR,
)


@dataclass(frozen=True)
class OptimizationResult:

    lambda_opt: np.ndarray

    elbo: float

    success: bool

    iterations: int


def optimize_reference(

    alpha,

    scipy_method,

    initial_lambda,

):

    initial_lambda = np.asarray(
        initial_lambda,
        dtype=np.float64,
    )

    bounds = [

        (LAMBDA_FLOOR, None)

        for _ in range(
            initial_lambda.size
        )

    ]

    result = minimize(

        fun=lambda lam: -dirichlet_elbo(
            alpha,
            lam,
            scipy_method,
        ),

        x0=initial_lambda,

        method="L-BFGS-B",

        bounds=bounds,

    )

    return OptimizationResult(

        lambda_opt=np.asarray(
            result.x,
            dtype=np.float64,
        ),

        elbo=float(
            -result.fun
        ),

        success=bool(
            result.success
        ),

        iterations=int(
            result.nit
        ),

    )


__all__ = [
    "OptimizationResult",
    "optimize_reference",
]