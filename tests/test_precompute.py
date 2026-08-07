"""
Tests for approximation precomputation utilities.
"""

import numpy as np

from approximation.model import (
    POLYNOMIAL_BERNSTEIN,
    POLYNOMIAL_RATIONAL,
    STIRLING_BERNSTEIN,
    STIRLING_RATIONAL,
)

from approximation.precompute import (
    build_cache,
)


MODELS = [
    POLYNOMIAL_BERNSTEIN,
    POLYNOMIAL_RATIONAL,
    STIRLING_BERNSTEIN,
    STIRLING_RATIONAL,
]


# ---------------------------------------------------------------------
# Basic cache tests
# ---------------------------------------------------------------------


def test_cache_shapes():

    x = np.linspace(
        1.0,
        2.0,
        100,
    )

    for model in MODELS:

        cache = build_cache(
            model,
            x,
        )

        assert cache.x.shape == (100,)

        assert cache.baseline_value.shape == (
            100,
        )

        assert cache.baseline_first.shape == (
            100,
        )

        assert cache.baseline_second.shape == (
            100,
        )

        assert cache.residual_basis.shape == (
            100,
            6,
        )

        assert cache.residual_first.shape == (
            100,
            6,
        )

        assert cache.residual_second.shape == (
            100,
            6,
        )


# ---------------------------------------------------------------------
# Value consistency
# ---------------------------------------------------------------------


def test_cached_value_matches_direct_evaluation():

    x = np.linspace(
        1.0,
        2.0,
        100,
    )

    coefficients = np.array(
        [
            0.2,
            0.4,
            0.1,
            0.8,
            0.5,
            0.3,
        ]
    )


    for model in MODELS:

        cache = build_cache(
            model,
            x,
        )


        direct = model(
            x,
            coefficients,
        )


        cached = (
            cache.baseline_value
            +
            cache.residual_basis
            @ coefficients
        )


        assert np.allclose(
            direct,
            cached,
        )


# ---------------------------------------------------------------------
# First derivative consistency
# ---------------------------------------------------------------------


def test_cached_first_matches_direct():

    x = np.linspace(
        1.0,
        2.0,
        100,
    )

    coefficients = np.array(
        [
            0.2,
            0.4,
            0.1,
            0.8,
            0.5,
            0.3,
        ]
    )


    for model in MODELS:

        cache = build_cache(
            model,
            x,
        )


        direct = model.first(
            x,
            coefficients,
        )


        cached = (
            cache.baseline_first
            +
            cache.residual_first
            @ coefficients
        )


        assert np.allclose(
            direct,
            cached,
        )


# ---------------------------------------------------------------------
# Second derivative consistency
# ---------------------------------------------------------------------


def test_cached_second_matches_direct():

    x = np.linspace(
        1.0,
        2.0,
        100,
    )

    coefficients = np.array(
        [
            0.2,
            0.4,
            0.1,
            0.8,
            0.5,
            0.3,
        ]
    )


    for model in MODELS:

        cache = build_cache(
            model,
            x,
        )


        direct = model.second(
            x,
            coefficients,
        )


        cached = (
            cache.baseline_second
            +
            cache.residual_second
            @ coefficients
        )


        assert np.allclose(
            direct,
            cached,
        )