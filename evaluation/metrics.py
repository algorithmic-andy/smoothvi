"""
Evaluation metrics for log-Gamma approximation models.

Metrics compare:

    G(x)      vs log Gamma(x)
    G'(x)     vs digamma(x)
    G''(x)    vs trigamma(x)

The module contains only metric calculations.
"""

from __future__ import annotations


import numpy as np

from scipy.special import (
    gammaln,
    digamma,
    polygamma,
)


from utils.routing import (
    route_to_core_interval,
)


EPSILON = 1e-12


# ============================================================
# Basic metrics
# ============================================================


def rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Root mean squared error.
    """

    prediction = np.asarray(prediction)
    target = np.asarray(target)

    return float(
        np.sqrt(
            np.mean(
                (prediction - target) ** 2
            )
        )
    )


def sd(
        prediction: np.ndarray,
        target: np.ndarray,
) -> float:
    """
    Prediction error standard deviation.
    """

    return float(
        np.std((prediction - target)**2, ddof=1)
    )


def max_error(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Maximum absolute error.
    """

    prediction = np.asarray(prediction)
    target = np.asarray(target)

    return float(
        np.max(
            np.abs(prediction - target)
        )
    )


def relative_error(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Pointwise relative error.
    """

    return (
        np.abs(prediction - target)
        /
        (
            np.abs(target)
            +
            EPSILON
        )
    )


def pearson_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Pearson correlation coefficient.
    """

    prediction = np.asarray(prediction)
    target = np.asarray(target)

    return float(
        np.corrcoef(
            prediction,
            target,
        )[0, 1]
    )


# ============================================================
# Routed approximation helper
# ============================================================


def routed_loggamma(
    model,
    coefficients,
    x: np.ndarray,
) -> np.ndarray:
    """
    Evaluate log-Gamma approximation using routing.

    Applies:

        logGamma(x)
        =
        sign * G(x_routed)
        + correction
    """

    routed = route_to_core_interval(
        x
    )

    return (
        routed.sign
        *
        model(
            routed.x,
            coefficients,
        )
        +
        routed.correction
    )


# ============================================================
# Gamma structure metrics
# ============================================================


def duplication_violation(
    model,
    coefficients,
    x: np.ndarray,
) -> np.ndarray:
    """
    Evaluate Legendre duplication identity violation.

    Identity:

        G(x)+G(x+1/2)
        =
        1/2 log(pi)
        +(1-2x)log(2)
        +G(2x)

    Every evaluation point is routed back into
    the approximation interval before evaluation.
    """


    lhs = (
        routed_loggamma(
            model,
            coefficients,
            x,
        )
        +
        routed_loggamma(
            model,
            coefficients,
            x + 0.5,
        )
    )


    rhs = (
        0.5 * np.log(np.pi)

        +
        (1.0 - 2.0 * x)
        *
        np.log(2.0)

        +
        routed_loggamma(
            model,
            coefficients,
            2.0 * x,
        )
    )


    return lhs - rhs



def duplication_rmse(
    model,
    coefficients,
    x: np.ndarray,
) -> float:
    """
    RMSE of duplication identity violation.
    """

    violation = duplication_violation(
        model,
        coefficients,
        x,
    )

    return float(
        np.sqrt(
            np.mean(
                violation ** 2
            )
        )
    )


# ============================================================
# Convexity diagnostics
# ============================================================


def convexity_violation_rate(
    second_derivative: np.ndarray,
) -> float:
    """
    Fraction of points violating log-convexity.

    Gamma is log-convex:

        d^2/dx^2 log Gamma(x) > 0
    """

    return float(
        np.mean(
            second_derivative < 0.0
        )
    )


# ============================================================
# Approximation Metrics
# ============================================================


def approximation_score(
        gamma_prediction: np.ndarray,
        gamma_true: np.ndarray,
        digamma_prediction: np.ndarray,
        digamma_true: np.ndarray,
        trigamma_prediction: np.ndarray,
        trigamma_true: np.ndarray,
) -> float:
    """
    Sum of gradient RMSE and standard deviation.
    """

    gamma_score = rmse(gamma_prediction, gamma_true)# + sd(gamma_prediction, gamma_true)
    digamma_score = rmse(digamma_prediction, digamma_true)# + sd(digamma_prediction, digamma_true)
    trigamma_score = rmse(trigamma_prediction, trigamma_true)# + sd(trigamma_prediction, trigamma_true)

    return float(
        gamma_score + digamma_score + trigamma_score
    )


# ============================================================
# Target functions
# ============================================================


def gamma_target(
    x: np.ndarray,
) -> np.ndarray:

    return gammaln(x)



def digamma_target(
    x: np.ndarray,
) -> np.ndarray:

    return digamma(x)



def trigamma_target(
    x: np.ndarray,
) -> np.ndarray:

    return polygamma(
        1,
        x,
    )


__all__ = [
    "rmse",
    "sd",
    "max_error",
    "relative_error",
    "pearson_correlation",

    "routed_loggamma",

    "duplication_violation",
    "duplication_rmse",

    "convexity_violation_rate",
    "approximation_score",

    "gamma_target",
    "digamma_target",
    "trigamma_target",
]