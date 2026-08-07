"""
Tests for combined log-Gamma approximation models.
"""

import numpy as np

from approximation.model import (
    POLYNOMIAL_BERNSTEIN,
    POLYNOMIAL_RATIONAL,
    STIRLING_BERNSTEIN,
    STIRLING_RATIONAL,
)


MODELS = [
    POLYNOMIAL_BERNSTEIN,
    POLYNOMIAL_RATIONAL,
    STIRLING_BERNSTEIN,
    STIRLING_RATIONAL,
]


def finite_difference_first(f, x, h=1e-6):
    return (
        f(x + h)
        - f(x - h)
    ) / (2 * h)


def finite_difference_second(f, x, h=1e-5):
    return (
        f(x + h)
        - 2 * f(x)
        + f(x - h)
    ) / h**2


# ---------------------------------------------------------------------
# Model existence
# ---------------------------------------------------------------------


def test_all_models_exist():

    assert len(MODELS) == 4

    identifiers = [
        model.identifier
        for model in MODELS
    ]

    assert len(set(identifiers)) == 4


# ---------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------


def test_model_output_shapes():

    x = np.linspace(1, 2, 100)

    coefficients = np.zeros(6)

    for model in MODELS:

        result = model(
            x,
            coefficients,
        )

        assert result.shape == x.shape



def test_zero_coefficients_equal_baseline():

    x = np.linspace(1, 2, 100)

    coefficients = np.zeros(6)

    for model in MODELS:

        approx = model(
            x,
            coefficients,
        )

        baseline = model.baseline(x)

        assert np.allclose(
            approx,
            baseline,
        )


# ---------------------------------------------------------------------
# Derivative tests
# ---------------------------------------------------------------------


def test_first_derivatives():

    x = 1.35

    coefficients = np.array(
        [
            0.2,
            0.4,
            0.1,
            0.8,
            0.5,
            0.3,
        ]
    )


    for model in MODELS:

        def f(z):
            return model(
                np.array([z]),
                coefficients,
            )[0]


        numerical = finite_difference_first(
            f,
            x,
        )

        analytic = model.first(
            np.array([x]),
            coefficients,
        )[0]


        assert np.isclose(
            numerical,
            analytic,
            rtol=1e-5,
            atol=1e-8,
        )


def test_second_derivatives():

    x = 1.35

    coefficients = np.array(
        [
            0.2,
            0.4,
            0.1,
            0.8,
            0.5,
            0.3,
        ]
    )


    for model in MODELS:

        def f(z):

            return model(
                np.array([z]),
                coefficients,
            )[0]


        numerical = finite_difference_second(
            f,
            x,
        )

        analytic = model.second(
            np.array([x]),
            coefficients,
        )[0]


        assert np.isclose(
            numerical,
            analytic,
            rtol=1e-4,
            atol=1e-6,
        )