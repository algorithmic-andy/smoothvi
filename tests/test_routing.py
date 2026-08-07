"""
Tests for Gamma routing.
"""

import numpy as np

from scipy.special import gammaln

from utils.routing import route_to_core_interval


def test_points_already_in_core_interval():
    """
    Points in [1,1.5] should remain unchanged.
    """

    x = np.array(
        [
            1.1,
            1.25,
            1.49,
        ]
    )

    result = route_to_core_interval(x)

    assert np.allclose(
        result.x,
        x,
    )

    assert np.allclose(
        result.correction,
        0.0,
    )

    assert np.all(
        result.sign == 1.0
    )


def test_large_values_use_recurrence():
    """
    Values above the fitting interval should be
    shifted downward.
    """

    x = np.array(
        [
            2.0,
            3.5,
            5.25,
        ]
    )

    result = route_to_core_interval(x)

    assert np.all(
        result.x >= 1.0
    )

    assert np.all(
        result.x <= 1.5
    )

    assert np.all(
        result.recurrence_depth > 0
    )


def test_reflection_branch_is_used():
    """
    Values in the upper half of [1,2]
    should reflect into [1,1.5].
    """

    x = np.array(
        [
            1.6,
            1.75,
            1.95,
        ]
    )

    result = route_to_core_interval(x)

    assert np.all(
        result.x >= 1.0
    )

    assert np.all(
        result.x <= 1.5
    )

    assert np.all(
        result.used_reflection
    )

    assert np.all(
        result.sign == -1.0
    )


def test_routing_preserves_log_gamma():
    """
    Full routing identity check against scipy.
    """

    x = np.linspace(
        0.2,
        10.0,
        500,
    )

    result = route_to_core_interval(x)

    lhs = gammaln(x)

    rhs = (
        result.sign
        * gammaln(result.x)
        + result.correction
    )

    assert np.allclose(
        lhs,
        rhs,
        atol=1e-12,
        rtol=1e-12,
    )