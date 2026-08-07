"""
Tests for approximation baselines.
"""

import numpy as np

from approximation.baselines import (
    POLYNOMIAL_BASELINE,
    STIRLING_BASELINE,
)


def finite_difference_first(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def finite_difference_second(f, x, h=1e-5):
    return (
        f(x + h)
        - 2 * f(x)
        + f(x - h)
    ) / (h ** 2)


def test_baseline_identifiers():
    assert POLYNOMIAL_BASELINE.identifier == "polynomial"
    assert STIRLING_BASELINE.identifier == "stirling"


def test_baseline_values_at_one():

    assert np.isclose(
        POLYNOMIAL_BASELINE(1.0),
        0.0,
    )

    assert np.isclose(
        STIRLING_BASELINE(1.0),
        0.0,
    )


def test_polynomial_endpoint_symmetry():

    assert np.isclose(
        POLYNOMIAL_BASELINE(2.0),
        0.0,
    )


def test_vectorized_evaluation():

    x = np.linspace(1, 2, 100)

    y_poly = POLYNOMIAL_BASELINE(x)
    y_stirling = STIRLING_BASELINE(x)

    assert y_poly.shape == x.shape
    assert y_stirling.shape == x.shape


def test_first_derivative():

    x = 1.35

    numerical = finite_difference_first(
        POLYNOMIAL_BASELINE,
        x,
    )

    analytic = POLYNOMIAL_BASELINE.first(x)

    assert np.isclose(
        numerical,
        analytic,
        rtol=1e-5,
    )


def test_second_derivative():

    x = 1.35

    numerical = finite_difference_second(
        POLYNOMIAL_BASELINE,
        x,
    )

    analytic = POLYNOMIAL_BASELINE.second(x)

    assert np.isclose(
        numerical,
        analytic,
        rtol=1e-4,
    )


def test_baselines_are_convex():

    x = np.linspace(1, 2, 200)

    assert np.all(
        POLYNOMIAL_BASELINE.second(x) > 0
    )

    assert np.all(
        STIRLING_BASELINE.second(x) > 0
    )