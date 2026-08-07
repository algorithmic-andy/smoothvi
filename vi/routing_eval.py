"""
Ultra-fast routing for Phase 2 evaluation.

Maps every positive x into [1,2] using only recurrence.

No reflection.
No Python recurrence loops.
"""

from __future__ import annotations

import numpy as np


# ==========================================================
# Helpers
# ==========================================================

def _broadcast_indices(max_steps: int):

    return np.arange(max_steps, dtype=np.float64)


# ==========================================================
# log Gamma
# ==========================================================

def route_log_gamma(x):

    x = np.asarray(x, dtype=np.float64)

    routed = x.copy()

    correction = np.zeros_like(routed)

    #
    # Downward recurrence
    #

    mask = routed > 2.0

    if np.any(mask):

        values = routed[mask]

        steps = np.floor(values - 2.0).astype(int) + 1

        K = np.max(steps)

        k = _broadcast_indices(K)

        terms = values[:, None] - 1.0 - k

        valid = k < steps[:, None]

        correction[mask] = np.sum(
            np.where(valid, np.log(terms), 0.0),
            axis=1,
        )

        routed[mask] -= steps

    #
    # Upward recurrence
    #

    mask = routed < 1.0

    if np.any(mask):

        values = routed[mask]

        steps = np.ceil(1.0 - values).astype(int)

        K = np.max(steps)

        k = _broadcast_indices(K)

        terms = values[:, None] + k

        valid = k < steps[:, None]

        correction[mask] -= np.sum(
            np.where(valid, np.log(terms), 0.0),
            axis=1,
        )

        routed[mask] += steps

    return routed, correction


# ==========================================================
# Digamma
# ==========================================================

def route_digamma(x):

    x = np.asarray(x, dtype=np.float64)

    routed = x.copy()

    correction = np.zeros_like(routed)

    mask = routed > 2.0

    if np.any(mask):

        values = routed[mask]

        steps = np.floor(values - 2.0).astype(int) + 1

        K = np.max(steps)

        k = _broadcast_indices(K)

        terms = values[:, None] - 1.0 - k

        valid = k < steps[:, None]

        correction[mask] = np.sum(
            np.where(valid, 1.0 / terms, 0.0),
            axis=1,
        )

        routed[mask] -= steps

    mask = routed < 1.0

    if np.any(mask):

        values = routed[mask]

        steps = np.ceil(1.0 - values).astype(int)

        K = np.max(steps)

        k = _broadcast_indices(K)

        terms = values[:, None] + k

        valid = k < steps[:, None]

        correction[mask] -= np.sum(
            np.where(valid, 1.0 / terms, 0.0),
            axis=1,
        )

        routed[mask] += steps

    return routed, correction


# ==========================================================
# Trigamma
# ==========================================================

def route_trigamma(x):

    x = np.asarray(x, dtype=np.float64)

    routed = x.copy()

    correction = np.zeros_like(routed)

    mask = routed > 2.0

    if np.any(mask):

        values = routed[mask]

        steps = np.floor(values - 2.0).astype(int) + 1

        K = np.max(steps)

        k = _broadcast_indices(K)

        terms = values[:, None] - 1.0 - k

        valid = k < steps[:, None]

        correction[mask] = -np.sum(
            np.where(valid, 1.0 / terms**2, 0.0),
            axis=1,
        )

        routed[mask] -= steps

    mask = routed < 1.0

    if np.any(mask):

        values = routed[mask]

        steps = np.ceil(1.0 - values).astype(int)

        K = np.max(steps)

        k = _broadcast_indices(K)

        terms = values[:, None] + k

        valid = k < steps[:, None]

        correction[mask] += np.sum(
            np.where(valid, 1.0 / terms**2, 0.0),
            axis=1,
        )

        routed[mask] += steps

    return routed, correction