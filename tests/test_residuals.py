"""
Tests for approximation residual families.
"""

import numpy as np

from approximation.residuals import (
    BERNSTEIN_RESIDUAL,
    RATIONAL_RESIDUAL,
)


# ---------------------------------------------------------------------
# Numerical differentiation helpers
# ---------------------------------------------------------------------

def finite_difference_first(f, x, h=1e-6):
    """
    Central finite difference first derivative.
    """
    return (
        f(x + h) - f(x - h)
    ) / (2 * h)


def finite_difference_second(f, x, h=1e-5):
    """
    Central finite difference second derivative.
    """
    return (
        f(x + h)
        - 2 * f(x)
        + f(x - h)
    ) / h**2


# ---------------------------------------------------------------------
# General tests
# ---------------------------------------------------------------------

def test_residual_identifiers():

    assert (
        BERNSTEIN_RESIDUAL.identifier
        == "bernstein"
    )

    assert (
        RATIONAL_RESIDUAL.identifier
        == "rational"
    )


def test_basis_shapes():

    x = np.linspace(1, 2, 50)

    bernstein = BERNSTEIN_RESIDUAL.basis(x)
    rational = RATIONAL_RESIDUAL.basis(x)

    assert bernstein.shape == (50, 6)

    assert rational.shape == (50, 6)


# ---------------------------------------------------------------------
# Bernstein tests
# ---------------------------------------------------------------------

def test_bernstein_partition_of_unity():

    x = np.linspace(1, 2, 100)

    basis = BERNSTEIN_RESIDUAL.basis(x)

    sums = basis.sum(axis=1)

    assert np.allclose(
        sums,
        1.0,
        rtol=1e-12,
    )


def test_bernstein_endpoint_behavior():

    left = BERNSTEIN_RESIDUAL.basis(
        np.array([1.0])
    )[0]

    right = BERNSTEIN_RESIDUAL.basis(
        np.array([2.0])
    )[0]


    # x=1 -> first basis only

    assert np.isclose(left[0], 1.0)

    assert np.allclose(
        left[1:],
        0.0,
    )


    # x=2 -> last basis only

    assert np.isclose(right[-1], 1.0)

    assert np.allclose(
        right[:-1],
        0.0,
    )


def test_bernstein_first_derivative():

    x = 1.35

    i = 2

    def basis_component(z):
        return (
            BERNSTEIN_RESIDUAL
            .basis(np.array([z]))[0, i]
        )


    numerical = finite_difference_first(
        basis_component,
        x,
    )

    analytic = (
        BERNSTEIN_RESIDUAL
        .first(np.array([x]))[0, i]
    )


    assert np.isclose(
        numerical,
        analytic,
        rtol=1e-5,
        atol=1e-8,
    )


def test_bernstein_second_derivative():

    x = 1.35

    i = 3

    def basis_component(z):
        return (
            BERNSTEIN_RESIDUAL
            .basis(np.array([z]))[0, i]
        )


    numerical = finite_difference_second(
        basis_component,
        x,
    )

    analytic = (
        BERNSTEIN_RESIDUAL
        .second(np.array([x]))[0, i]
    )


    assert np.isclose(
        numerical,
        analytic,
        rtol=1e-4,
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# Rational tests
# ---------------------------------------------------------------------

def test_rational_vanishes_at_one():

    basis = RATIONAL_RESIDUAL.basis(
        np.array([1.0])
    )[0]

    assert np.allclose(
        basis,
        0.0,
    )


def test_rational_first_derivative():

    x = 1.35

    i = 3


    def basis_component(z):
        return (
            RATIONAL_RESIDUAL
            .basis(np.array([z]))[0, i]
        )


    numerical = finite_difference_first(
        basis_component,
        x,
    )

    analytic = (
        RATIONAL_RESIDUAL
        .first(np.array([x]))[0, i]
    )


    assert np.isclose(
        numerical,
        analytic,
        rtol=1e-5,
        atol=1e-8,
    )


def test_rational_second_derivative():

    x = 1.35

    i = 4


    def basis_component(z):
        return (
            RATIONAL_RESIDUAL
            .basis(np.array([z]))[0, i]
        )


    numerical = finite_difference_second(
        basis_component,
        x,
    )

    analytic = (
        RATIONAL_RESIDUAL
        .second(np.array([x]))[0, i]
    )


    assert np.isclose(
        numerical,
        analytic,
        rtol=1e-4,
        atol=1e-6,
    )


def test_rational_convexity():

    x = np.linspace(1, 2, 200)

    second = RATIONAL_RESIDUAL.second(x)

    # Every basis function should have non-negative curvature
    assert np.all(second >= 0)