"""
Reflection transforms for Gamma related functions.

Implements Euler reflection identities for:

log-Gamma
Digamma
Trigamma
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


LOG_PI = np.log(np.pi)


@dataclass(frozen=True)
class ReflectionResult:

    x: NDArray[np.float64]
    correction: NDArray[np.float64]


# ==============================================================
# log-Gamma reflection
# ==============================================================


def apply_reflection(
    x: ArrayLike,
) -> ReflectionResult:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    reflected = 1.0 - x

    correction = (
        LOG_PI
        -
        np.log(
            np.sin(np.pi*x)
        )
    )

    return ReflectionResult(
        x=reflected,
        correction=correction,
    )


# ==============================================================
# Digamma reflection
# ==============================================================


def apply_digamma_reflection(
    x: ArrayLike,
) -> ReflectionResult:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    reflected = 1.0 - x

    correction = (
        -np.pi
        *
        np.cos(np.pi*x)
        /
        np.sin(np.pi*x)
    )

    return ReflectionResult(
        x=reflected,
        correction=correction,
    )


# ==============================================================
# Trigamma reflection
# ==============================================================


def apply_trigamma_reflection(
    x: ArrayLike,
) -> ReflectionResult:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    reflected = 1.0 - x

    correction = (
        np.pi**2
        /
        np.sin(np.pi*x)**2
    )

    return ReflectionResult(
        x=reflected,
        correction=correction,
    )


__all__ = [
    "ReflectionResult",
    "apply_reflection",
    "apply_digamma_reflection",
    "apply_trigamma_reflection",
]