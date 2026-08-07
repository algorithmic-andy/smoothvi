"""
Utilities for generating Differential Evolution populations.

A single initial population can be reused across optimization runs
to reduce randomness when comparing different approximation models
or fitting objectives.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from utils.config import DE_POPSIZE, DE_SEED



def generate_initial_population(
    num_coefficients: int,
    population_size: int = DE_POPSIZE,
    seed: int = DE_SEED,
) -> NDArray[np.float64]:
    """
    Generate a reproducible Differential Evolution population.

    Parameters
    ----------
    num_coefficients
        Number of coefficients being optimized.

    population_size
        Number of candidate solutions.

    seed
        Random seed.

    Returns
    -------
    population
        Array of shape:

            (population_size, num_coefficients)

        with all coefficients in [0,1].
    """

    rng = np.random.default_rng(
        seed
    )

    return rng.uniform(
        low=0.0,
        high=1.0,
        size=(
            population_size,
            num_coefficients,
        ),
    )


__all__ = [
    "generate_initial_population",
]