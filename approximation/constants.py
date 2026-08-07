"""
constants.py

Mathematical constants and approximation defaults.

This module contains fixed values used throughout the approximation
package, including approximation intervals, basis information, and
mathematical constants.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Approximation domains
# ============================================================

# Interval used during coefficient fitting.
#
# Values in [1.5, 2] are reflected back into this interval using
# Euler reflection when constructing the fitting objective.
FIT_INTERVAL = (1.0, 1.5)


# Interval where the final approximation is valid.
#
# Values outside this interval are handled through recurrence.
APPROXIMATION_INTERVAL = (1.0, 2.0)


# Interval where extrapolation is valid.
#
# Values in this interval route to the approximation interval.
EXTRAPOLATION_INTERVAL = (0.1, 10.0)


# ============================================================
# Approximation structure
# ============================================================

# All residual families use six coefficients:
#
# c_0, ..., c_5
NUM_COEFFICIENTS = 6


# Degree of the Bernstein polynomial residual.
#
# Six coefficients correspond to degree five.
POLYNOMIAL_DEGREE = NUM_COEFFICIENTS - 1


# ============================================================
# Bernstein basis constants
# ============================================================

# Binomial coefficients for the degree-5 Bernstein basis:
#
# [1, 5, 10, 10, 5, 1]
BERNSTEIN_BINOMIAL = np.array(
    [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
    dtype=np.float64,
)


# ============================================================
# Mathematical constants
# ============================================================

LOG_PI = np.log(np.pi)

LOG_2 = np.log(2.0)

LOG_SQRT_PI = 0.5 * LOG_PI

SQRT_PI = np.sqrt(np.pi)

LOG_SQRT_PI_OVER_TWO = LOG_SQRT_PI - LOG_2


# ============================================================
# Baseline constants
# ============================================================

# Polynomial Baseline:
#
# B_P(x)
# =
# -4 log(sqrt(pi)/2)(x-1)(x-2)
#
# We store the leading multiplicative constant separately.
POLYNOMIAL_BASELINE_SCALE = -4.0 * LOG_SQRT_PI_OVER_TWO



__all__ = [
    "FIT_INTERVAL",
    "APPROXIMATION_INTERVAL",
    "NUM_COEFFICIENTS",
    "POLYNOMIAL_DEGREE",
    "BERNSTEIN_BINOMIAL",
    "LOG_PI",
    "LOG_2",
    "LOG_SQRT_PI",
    "SQRT_PI",
    "LOG_SQRT_PI_OVER_TWO",
    "POLYNOMIAL_BASELINE_SCALE",
]