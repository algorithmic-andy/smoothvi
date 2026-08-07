"""
Residual correction functions for log-Gamma approximation.

This module defines residual families used to augment baseline
approximations of log(Gamma(x)).

Each residual provides:
- basis functions
- first derivatives
- second derivatives

The residuals are represented as linear combinations:

    R(x) = Phi(x) @ c

where Phi(x) is the residual basis matrix and c is the coefficient
vector.

Available residuals
-------------------
- Bernstein Residual
- Rational Residual
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    BERNSTEIN_BINOMIAL,
    NUM_COEFFICIENTS,
    POLYNOMIAL_DEGREE,
)


@dataclass(frozen=True)
class Residual:
    """
    Mathematical residual correction family.

    Parameters
    ----------
    identifier
        Unique identifier for serialization.

    display_name
        Human-readable name for plots/tables.

    basis
        Function returning basis matrix.

    first
        Function returning first derivative basis matrix.

    second
        Function returning second derivative basis matrix.

    degree
        Polynomial degree (if applicable).

    num_coefficients
        Number of coefficients in the residual expansion.
    """

    identifier: str
    display_name: str

    basis: Callable[[ArrayLike], NDArray[np.float64]]
    first: Callable[[ArrayLike], NDArray[np.float64]]
    second: Callable[[ArrayLike], NDArray[np.float64]]

    degree: int
    num_coefficients: int

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the residual basis."""
        return self.basis(x)


# =====================================================================
# Bernstein Residual
# =====================================================================


def _bernstein_basis(x: ArrayLike) -> NDArray[np.float64]:
    """
    Evaluate degree-5 Bernstein basis on [1,2].

    Uses transformed coordinate:

        t = x - 1

    so that t in [0,1].
    """

    x = np.asarray(x, dtype=np.float64)

    t = x - 1.0
    one_minus_t = 1.0 - t

    basis = np.empty(
        (t.size, NUM_COEFFICIENTS),
        dtype=np.float64,
    )

    for i in range(NUM_COEFFICIENTS):
        basis[:, i] = (
            BERNSTEIN_BINOMIAL[i]
            * t**i
            * one_minus_t ** (POLYNOMIAL_DEGREE - i)
        )

    return basis


def _bernstein_first(x: ArrayLike) -> NDArray[np.float64]:
    """
    First derivative of Bernstein basis.

    Uses the identity:

        b'_{i,n}
        =
        n(b_{i-1,n-1} - b_{i,n-1})
    """

    x = np.asarray(x, dtype=np.float64)

    t = x - 1.0

    lower_degree = np.empty(
        (t.size, POLYNOMIAL_DEGREE),
        dtype=np.float64,
    )

    for i in range(POLYNOMIAL_DEGREE):
        lower_degree[:, i] = (
            math.comb(POLYNOMIAL_DEGREE - 1, i)
            * t**i
            * (1 - t) ** (
                POLYNOMIAL_DEGREE - 1 - i
            )
        )

    derivative = np.zeros(
        (t.size, NUM_COEFFICIENTS),
        dtype=np.float64,
    )

    for i in range(NUM_COEFFICIENTS):
        left = lower_degree[:, i - 1] if i > 0 else 0.0
        right = (
            lower_degree[:, i]
            if i < POLYNOMIAL_DEGREE
            else 0.0
        )

        derivative[:, i] = POLYNOMIAL_DEGREE * (
            left - right
        )

    return derivative


def _bernstein_second(x: ArrayLike) -> NDArray[np.float64]:
    """
    Second derivative of Bernstein basis.
    """

    x = np.asarray(x, dtype=np.float64)

    t = x - 1.0

    degree_two = np.empty(
        (t.size, POLYNOMIAL_DEGREE - 1),
        dtype=np.float64,
    )

    for i in range(POLYNOMIAL_DEGREE - 1):
        degree_two[:, i] = (
            math.comb(POLYNOMIAL_DEGREE - 2, i)
            * t**i
            * (1 - t) ** (
                POLYNOMIAL_DEGREE - 2 - i
            )
        )

    derivative = np.zeros(
        (t.size, NUM_COEFFICIENTS),
        dtype=np.float64,
    )

    n = POLYNOMIAL_DEGREE

    for i in range(NUM_COEFFICIENTS):

        term = 0.0

        if i >= 2:
            term += degree_two[:, i - 2]

        if 1 <= i <= n - 1:
            term -= 2.0 * degree_two[:, i - 1]

        if i <= n - 2:
            term += degree_two[:, i]

        derivative[:, i] = n * (n - 1) * term

    return derivative


# =====================================================================
# Rational Residual
# =====================================================================


def _rational_basis(x: ArrayLike) -> NDArray[np.float64]:
    """
    Rational residual basis.

    Basis:

        phi_0(x) = (x-1)^2

        phi_i(x) = (x-1)^2/(x+i-1)

        i=1,...,5
    """

    x = np.asarray(x, dtype=np.float64)

    y = x - 1.0

    basis = np.empty(
        (x.size, NUM_COEFFICIENTS),
        dtype=np.float64,
    )

    basis[:, 0] = y**2

    for i in range(1, NUM_COEFFICIENTS):
        basis[:, i] = y**2 / (x + i - 1)

    return basis


def _rational_first(x: ArrayLike) -> NDArray[np.float64]:

    x = np.asarray(x, dtype=np.float64)

    y = x - 1.0

    derivative = np.empty(
        (x.size, NUM_COEFFICIENTS),
        dtype=np.float64,
    )

    derivative[:, 0] = 2.0 * y

    for i in range(1, NUM_COEFFICIENTS):

        denom = x + i - 1

        derivative[:, i] = (
            2.0 * y / denom
            - y**2 / denom**2
        )

    return derivative


def _rational_second(x: ArrayLike) -> NDArray[np.float64]:

    x = np.asarray(x, dtype=np.float64)

    y = x - 1.0

    derivative = np.empty(
        (x.size, NUM_COEFFICIENTS),
        dtype=np.float64,
    )

    derivative[:, 0] = 2.0

    for i in range(1, NUM_COEFFICIENTS):

        denom = x + i - 1

        derivative[:, i] = (
            2.0 / denom
            - 4.0 * y / denom**2
            + 2.0 * y**2 / denom**3
        )

    return derivative


# =====================================================================
# Public residual objects
# =====================================================================

BERNSTEIN_RESIDUAL = Residual(
    identifier="bernstein",
    display_name="Bernstein Residual",
    basis=_bernstein_basis,
    first=_bernstein_first,
    second=_bernstein_second,
    degree=POLYNOMIAL_DEGREE,
    num_coefficients=NUM_COEFFICIENTS,
)


RATIONAL_RESIDUAL = Residual(
    identifier="rational",
    display_name="Rational Residual",
    basis=_rational_basis,
    first=_rational_first,
    second=_rational_second,
    degree=-1,
    num_coefficients=NUM_COEFFICIENTS,
)


__all__ = [
    "Residual",
    "BERNSTEIN_RESIDUAL",
    "RATIONAL_RESIDUAL",
]