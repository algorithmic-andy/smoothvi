"""
Tests for coefficient optimization.
"""

from __future__ import annotations

import numpy as np

from scipy.special import gammaln

from approximation.model import (
    POLYNOMIAL_BERNSTEIN,
)

from approximation.precompute import (
    build_cache,
)

from fitting.losses import (
    LEAST_SQUARES_LOSS,
)

from fitting.optimizer import (
    fit_coefficients,
)

from fitting.population import (
    generate_initial_population,
)



def test_generate_population_bounds():
    """
    Initial population should lie inside [0,1].
    """

    population = generate_initial_population(
        num_coefficients=6,
        population_size=10,
        seed=42,
    )


    assert population.shape == (
        10,
        6,
    )

    assert np.all(
        population >= 0.0
    )

    assert np.all(
        population <= 1.0
    )



def test_optimizer_returns_valid_coefficients():
    """
    Differential evolution should return bounded coefficients.
    """

    x = np.linspace(
        1.0,
        2.0,
        30,
    )

    cache = build_cache(
        POLYNOMIAL_BERNSTEIN,
        x,
    )

    target = gammaln(
        x
    )


    population = generate_initial_population(
        num_coefficients=6,
        population_size=8,
    )


    result = fit_coefficients(
        loss=LEAST_SQUARES_LOSS,
        objective_args={
            "cache": cache,
            "target": target,
        },
        num_coefficients=6,
        initial_population=population,
    )


    assert result.coefficients.shape == (
        6,
    )

    assert np.all(
        result.coefficients >= 0.0
    )

    assert np.all(
        result.coefficients <= 1.0
    )

    assert np.isfinite(
        result.loss
    )



def test_same_population_reproducibility():
    """
    Same initial population should produce deterministic results
    under a fixed DE seed.
    """

    population = generate_initial_population(
        num_coefficients=6,
        population_size=8,
        seed=123,
    )


    assert np.allclose(
        population,
        generate_initial_population(
            num_coefficients=6,
            population_size=8,
            seed=123,
        ),
    )