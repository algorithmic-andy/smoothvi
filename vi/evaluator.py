"""
Fast evaluator used by ApproximationMethod.

This replaces hundreds of nested closures with
three lightweight vectorized functions.
"""

from __future__ import annotations

import numpy as np

from .routing_eval import (
    route_log_gamma,
    route_digamma,
    route_trigamma,
)

from .digamma import (
    digamma_core,
    trigamma_core,
)


def evaluate_log_gamma(cache, x):

    x, correction = route_log_gamma(x)    

    value = (
        cache.baseline.value(x)
        + cache.residual.basis(x) @ cache.coefficients
        + correction
    )

    return value


def evaluate_digamma(cache, x):

    

    if cache.use_series:

        x, correction = route_digamma(x)        

        value = (
            digamma_core(x) + correction
        )

        return value

    x, correction = route_digamma(x)    

    value = (
        cache.baseline.first(x)
        + cache.residual.first(x)
        @ cache.coefficients
        + correction
    )

    return value


def evaluate_trigamma(cache, x):

    

    if cache.use_series:

        x, correction = route_trigamma(x)       

        value = (
            trigamma_core(x) + correction
        )

        return value
    
    x, correction = route_trigamma(x)   

    value = (
        cache.baseline.second(x)
        + cache.residual.second(x)
        @ cache.coefficients
        + correction
    )

    return value