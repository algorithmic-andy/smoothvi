"""
Direct Digamma and Trigamma approximations.

These functions evaluate the approximation ONLY on the core interval.

Routing is handled separately by approximations.py.
"""

from __future__ import annotations

import numpy as np


def digamma_core(x):
    """
    Sixth-order asymptotic Digamma approximation
    on the routed interval.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return (

        np.log(x)

        - 1.0 / (2.0 * x)

        - 1.0 / (12.0 * x**2)

        + 1.0 / (120.0 * x**4)

        - 1.0 / (252.0 * x**6)

    )


def trigamma_core(x):
    """
    Analytic derivative of digamma_core().
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return (

        1.0 / x

        + 1.0 / (2.0 * x**2)

        + 1.0 / (6.0 * x**3)

        - 1.0 / (30.0 * x**5)

        + 1.0 / (42.0 * x**7)

    )


__all__ = [
    "digamma_core",
    "trigamma_core",
]