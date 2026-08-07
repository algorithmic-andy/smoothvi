"""
Tests for fitting loss functions.

These tests verify:

- approximation evaluation from caches
- least squares loss behavior
- duplication identity loss behavior
"""

from __future__ import annotations

import numpy as np

from scipy.special import gammaln

from approximation.model import (
    POLYNOMIAL_BERNSTEIN,
)

from approximation.precompute import (
    build_cache,
)

from fitting.losses import (
    LEAST_SQUARES_LOSS,
    DUPLICATION_LOSS,
)

from utils.routing import (
    route_to_core_interval,
)



def test_least_squares_loss_finite():
    """
    Least squares loss should return a finite value.
    """

    x = np.linspace(
        1.0,
        2.0,
        50,
    )

    cache = build_cache(
        POLYNOMIAL_BERNSTEIN,
        x,
    )

    target = gammaln(
        x
    )

    coefficients = np.full(
        6,
        0.5,
    )


    value = LEAST_SQUARES_LOSS.evaluate(
        cache=cache,
        target=target,
        coefficients=coefficients,
    )


    assert np.isfinite(
        value
    )

    assert value >= 0.0



def test_least_squares_zero_error_for_exact_target():
    """
    If the approximation exactly equals the target,
    the least squares loss should vanish.
    """

    x = np.linspace(
        1.0,
        2.0,
        20,
    )

    cache = build_cache(
        POLYNOMIAL_BERNSTEIN,
        x,
    )

    target = (
        cache.baseline_value
    )

    coefficients = np.zeros(
        6,
    )


    value = LEAST_SQUARES_LOSS.evaluate(
        cache=cache,
        target=target,
        coefficients=coefficients,
    )


    assert np.isclose(
        value,
        0.0,
    )



def test_duplication_loss_finite():
    """
    Duplication loss should evaluate without errors.
    """

    x = np.linspace(
        1.0,
        2.0,
        50,
    )


    routed_x = route_to_core_interval(
        x,
    )

    routed_half = route_to_core_interval(
        x + 0.5,
    )

    routed_two = route_to_core_interval(
        2.0 * x,
    )


    cache_x = build_cache(
        POLYNOMIAL_BERNSTEIN,
        routed_x.x,
        routed_x.correction,
        routed_x.sign,
    )

    cache_half = build_cache(
        POLYNOMIAL_BERNSTEIN,
        routed_half.x,
        routed_half.correction,
        routed_half.sign,
    )

    cache_two = build_cache(
        POLYNOMIAL_BERNSTEIN,
        routed_two.x,
        routed_two.correction,
        routed_two.sign,
    )


    coefficients = np.full(
        6,
        0.5,
    )


    value = DUPLICATION_LOSS.evaluate(
        cache_x=cache_x,
        cache_half=cache_half,
        cache_two=cache_two,
        coefficients=coefficients,
    )


    assert np.isfinite(
        value
    )

    assert value >= 0.0