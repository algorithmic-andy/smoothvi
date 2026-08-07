"""
Recurrence routing utilities.

This module implements recurrence transport for:

1. log-Gamma
2. Digamma
3. Trigamma

All functions transport values into a target interval while
returning the additive correction required to recover the original
quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ==============================================================
# Result containers
# ==============================================================


@dataclass(frozen=True)
class RecurrenceResult:
    """
    Result of log-Gamma recurrence routing.
    """

    x: NDArray[np.float64]
    correction: NDArray[np.float64]


@dataclass(frozen=True)
class DerivativeRecurrenceResult:
    """
    Result of derivative recurrence routing.

    Used for Digamma and Trigamma.
    """

    x: NDArray[np.float64]
    correction: NDArray[np.float64]


# ==============================================================
# Log-Gamma recurrence
# ==============================================================


def apply_recurrence(
    x: ArrayLike,
    interval: tuple[float, float],
) -> RecurrenceResult:

    lower, upper = interval

    if lower >= upper:
        raise ValueError(
            "Interval must satisfy lower < upper."
        )

    shifted = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(
        shifted,
        dtype=np.float64,
    )

    # x -> x-1
    while True:

        mask = shifted > upper

        if not np.any(mask):
            break

        correction[mask] += np.log(
            shifted[mask] - 1.0
        )

        shifted[mask] -= 1.0


    # x -> x+1
    while True:

        mask = shifted < lower

        if not np.any(mask):
            break

        correction[mask] -= np.log(
            shifted[mask]
        )

        shifted[mask] += 1.0


    return RecurrenceResult(
        x=shifted,
        correction=correction,
    )


# ==============================================================
# Digamma recurrence
# ==============================================================


def apply_digamma_recurrence(
    x: ArrayLike,
    interval: tuple[float, float],
) -> DerivativeRecurrenceResult:

    lower, upper = interval

    shifted = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(
        shifted,
        dtype=np.float64,
    )


    # psi(x)=psi(x-1)+1/(x-1)
    while True:

        mask = shifted > upper

        if not np.any(mask):
            break

        correction[mask] += (
            1.0 / (shifted[mask] - 1.0)
        )

        shifted[mask] -= 1.0


    # psi(x)=psi(x+1)-1/x
    while True:

        mask = shifted < lower

        if not np.any(mask):
            break

        correction[mask] -= (
            1.0 / shifted[mask]
        )

        shifted[mask] += 1.0


    return DerivativeRecurrenceResult(
        x=shifted,
        correction=correction,
    )


# ==============================================================
# Trigamma recurrence
# ==============================================================


def apply_trigamma_recurrence(
    x: ArrayLike,
    interval: tuple[float, float],
) -> DerivativeRecurrenceResult:

    lower, upper = interval

    shifted = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(
        shifted,
        dtype=np.float64,
    )


    # psi'(x)=psi'(x-1)-1/(x-1)^2
    while True:

        mask = shifted > upper

        if not np.any(mask):
            break

        correction[mask] -= (
            1.0 /
            (shifted[mask] - 1.0)**2
        )

        shifted[mask] -= 1.0


    # psi'(x)=psi'(x+1)+1/x^2
    while True:

        mask = shifted < lower

        if not np.any(mask):
            break

        correction[mask] += (
            1.0 /
            shifted[mask]**2
        )

        shifted[mask] += 1.0


    return DerivativeRecurrenceResult(
        x=shifted,
        correction=correction,
    )


__all__ = [
    "RecurrenceResult",
    "DerivativeRecurrenceResult",
    "apply_recurrence",
    "apply_digamma_recurrence",
    "apply_trigamma_recurrence",
]