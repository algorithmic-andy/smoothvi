"""
Gamma-family routing utilities.

This module defines routing strategies for:

1. log-Gamma
2. Digamma
3. Trigamma

All functions transport arbitrary positive inputs into the
core approximation interval [1, 1.5].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reflection import (
    apply_reflection,
    apply_digamma_reflection,
    apply_trigamma_reflection,
)


# ==============================================================
# Result containers
# ==============================================================


@dataclass(frozen=True)
class RoutingResult:
    """
    Routing result for log-Gamma.
    """

    x: NDArray[np.float64]
    correction: NDArray[np.float64]
    sign: NDArray[np.float64]
    recurrence_depth: NDArray[np.int64]
    used_reflection: NDArray[np.bool_]


@dataclass(frozen=True)
class DerivativeRoutingResult:
    """
    Routing result for Digamma/Trigamma.
    """

    x: NDArray[np.float64]
    correction: NDArray[np.float64]


# ==============================================================
# log-Gamma routing
# ==============================================================


def route_to_core_interval(
    x: ArrayLike,
    lower: float = 1.0,
    upper: float = 1.5,
) -> RoutingResult:

    routed = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(routed)
    sign = np.ones_like(routed)

    recurrence_depth = np.zeros_like(
        routed,
        dtype=np.int64,
    )

    used_reflection = np.zeros_like(
        routed,
        dtype=bool,
    )


    # x -> x-1
    while True:

        mask = routed > upper

        if not np.any(mask):
            break

        correction[mask] += np.log(
            routed[mask] - 1.0
        )

        routed[mask] -= 1.0

        recurrence_depth[mask] += 1


    # x -> x+1
    while True:

        mask = routed < 0.5

        if not np.any(mask):
            break

        correction[mask] -= np.log(
            routed[mask]
        )

        routed[mask] += 1.0

        recurrence_depth[mask] += 1


    # Reflection branch
    reflect_mask = (
        (routed >= 0.5)
        &
        (routed < 1.0)
    )

    if np.any(reflect_mask):

        reflected = apply_reflection(
            routed[reflect_mask]
        )

        correction[reflect_mask] += (
            reflected.correction
        )

        routed[reflect_mask] = reflected.x

        sign[reflect_mask] = -1.0

        used_reflection[reflect_mask] = True


        # reflected y -> y+1

        correction[reflect_mask] += np.log(
            routed[reflect_mask]
        )

        routed[reflect_mask] += 1.0

        recurrence_depth[reflect_mask] += 1

    return RoutingResult(
        routed,
        correction,
        sign,
        recurrence_depth,
        used_reflection,
    )

# ==============================================================
# log-Gamma evaluation routing
# ==============================================================


def route_to_eval_interval(
    x: ArrayLike,
    lower: float = 1.0,
    upper: float = 2.0,
) -> RoutingResult:
    """
    Route positive inputs into the evaluation interval [1, 2]
    using recurrence only.

    No reflection is used.
    """

    routed = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(routed)

    sign = np.ones_like(routed)

    recurrence_depth = np.zeros_like(
        routed,
        dtype=np.int64,
    )

    used_reflection = np.zeros_like(
        routed,
        dtype=bool,
    )

    while True:

        mask = routed > upper

        if not np.any(mask):
            break

        correction[mask] += np.log(
            routed[mask] - 1.0
        )

        routed[mask] -= 1.0

        recurrence_depth[mask] += 1

    while True:

        mask = routed < lower

        if not np.any(mask):
            break

        correction[mask] -= np.log(
            routed[mask]
        )

        routed[mask] += 1.0

        recurrence_depth[mask] += 1

    return RoutingResult(
        x=routed,
        correction=correction,
        sign=sign,
        recurrence_depth=recurrence_depth,
        used_reflection=used_reflection,
    )


# ==============================================================
# Digamma routing
# ==============================================================


def route_digamma_to_core_interval(
    x: ArrayLike,
    lower: float = 1.0,
    upper: float = 1.5,
) -> DerivativeRoutingResult:


    routed = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(routed)


    # psi(x)=psi(x-1)+1/(x-1)

    while True:

        mask = routed > upper

        if not np.any(mask):
            break

        correction[mask] += (
            1.0 /
            (routed[mask]-1.0)
        )

        routed[mask] -= 1.0


    # psi(x)=psi(x+1)-1/x

    while True:

        mask = routed < 0.5

        if not np.any(mask):
            break

        correction[mask] -= (
            1.0 /
            routed[mask]
        )

        routed[mask] += 1.0


    # Reflection

    reflect_mask = (
        (routed >= 0.5)
        &
        (routed < 1.0)
    )

    if np.any(reflect_mask):

        reflected = apply_digamma_reflection(
            routed[reflect_mask]
        )

        correction[reflect_mask] += (
            reflected.correction
        )

        routed[reflect_mask] = reflected.x


        # recurrence after reflection

        correction[reflect_mask] -= (
            1.0 /
            routed[reflect_mask]
        )

        routed[reflect_mask] += 1.0

    return DerivativeRoutingResult(
        routed,
        correction,
    )

# ==============================================================
# Digamma evaluation routing
# ==============================================================


def route_digamma_to_eval_interval(
    x: ArrayLike,
    lower: float = 1.0,
    upper: float = 2.0,
) -> DerivativeRoutingResult:
    """
    Route Digamma inputs into [1, 2]
    using recurrence only.
    """

    routed = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(routed)

    while True:

        mask = routed > upper

        if not np.any(mask):
            break

        correction[mask] += (
            1.0 /
            (routed[mask] - 1.0)
        )

        routed[mask] -= 1.0

    while True:

        mask = routed < lower

        if not np.any(mask):
            break

        correction[mask] -= (
            1.0 /
            routed[mask]
        )

        routed[mask] += 1.0

    return DerivativeRoutingResult(
        x=routed,
        correction=correction,
    )


# ==============================================================
# Trigamma routing
# ==============================================================


def route_trigamma_to_core_interval(
    x: ArrayLike,
    lower: float = 1.0,
    upper: float = 1.5,
) -> DerivativeRoutingResult:


    routed = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(routed)


    # psi'(x)=psi'(x-1)-1/(x-1)^2

    while True:

        mask = routed > upper

        if not np.any(mask):
            break

        correction[mask] -= (
            1.0 /
            (routed[mask]-1.0)**2
        )

        routed[mask] -= 1.0


    # psi'(x)=psi'(x+1)+1/x^2

    while True:

        mask = routed < 0.5

        if not np.any(mask):
            break

        correction[mask] += (
            1.0 /
            routed[mask]**2
        )

        routed[mask] += 1.0


    # Reflection

    reflect_mask = (
        (routed >= 0.5)
        &
        (routed < 1.0)
    )

    if np.any(reflect_mask):

        reflected = apply_trigamma_reflection(
            routed[reflect_mask]
        )

        correction[reflect_mask] += (
            reflected.correction
        )

        routed[reflect_mask] = reflected.x


        # no sign flip for trigamma
        # recurrence correction after reflection

        correction[reflect_mask] -= (
            1.0 /
            routed[reflect_mask]**2
        )

        routed[reflect_mask] += 1.0

    return DerivativeRoutingResult(
        routed,
        correction,
    )

# ==============================================================
# Trigamma evaluation routing
# ==============================================================


def route_trigamma_to_eval_interval(
    x: ArrayLike,
    lower: float = 1.0,
    upper: float = 2.0,
) -> DerivativeRoutingResult:
    """
    Route Trigamma inputs into [1, 2]
    using recurrence only.
    """

    routed = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    correction = np.zeros_like(routed)

    while True:

        mask = routed > upper

        if not np.any(mask):
            break

        correction[mask] -= (
            1.0 /
            (routed[mask] - 1.0) ** 2
        )

        routed[mask] -= 1.0

    while True:

        mask = routed < lower

        if not np.any(mask):
            break

        correction[mask] += (
            1.0 /
            routed[mask] ** 2
        )

        routed[mask] += 1.0

    return DerivativeRoutingResult(
        x=routed,
        correction=correction,
    )

__all__ = [
    "RoutingResult",
    "DerivativeRoutingResult",
    "route_to_core_interval",
    "route_digamma_to_core_interval",
    "route_trigamma_to_core_interval",
]