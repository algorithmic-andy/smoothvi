"""
Loads every frozen approximation used during Phase 2.

Each approximation exposes

    log_gamma(x)
    digamma(x)
    trigamma(x)

using cached evaluators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scipy.special import (
    gammaln,
    digamma,
    polygamma,
)

from approximation.model import ALL_APPROXIMATIONS

from .cache import ApproximationCache
from .evaluator import (
    evaluate_log_gamma,
    evaluate_digamma,
    evaluate_trigamma,
)

COEFFICIENT_FILE = Path(
    "results/fitted_coefficients.json"
)

MODEL_LOOKUP = {
    model.identifier: model
    for model in ALL_APPROXIMATIONS
}


@dataclass(frozen=True)
class ApproximationMethod:

    identifier: str

    display_name: str

    cache: ApproximationCache | None = None

    exact: bool = False

    def log_gamma(self, x):

        if self.exact:
            return gammaln(x)

        return evaluate_log_gamma(
            self.cache,
            x,
        )

    def digamma(self, x):

        if self.exact:
            return digamma(x)

        return evaluate_digamma(
            self.cache,
            x,
        )

    def trigamma(self, x):

        if self.exact:
            return polygamma(1, x)

        return evaluate_trigamma(
            self.cache,
            x,
        )


def _load_coefficients():

    with open(COEFFICIENT_FILE) as f:
        return json.load(f)


def build_methods():

    methods = []

    fitted = _load_coefficients()

    for result in fitted:

        model = MODEL_LOOKUP[
            result["model_id"]
        ]

        coeffs = np.asarray(
            result["coefficients"],
            dtype=np.float64,
        )

        ####################################################
        # Derivative
        ####################################################

        methods.append(

            ApproximationMethod(

                identifier=(
                    f"{result['model_id']}_"
                    f"{result['loss_id']}_"
                    "derivative"
                ),

                display_name=(
                    f"{model.display_name}"
                    " (Derivative)"
                ),

                cache=ApproximationCache(

                    baseline=model.baseline,

                    residual=model.residual,

                    coefficients=coeffs,

                    use_series=False,

                ),
            )
        )

        ####################################################
        # Series
        ####################################################

        methods.append(

            ApproximationMethod(

                identifier=(
                    f"{result['model_id']}_"
                    f"{result['loss_id']}_"
                    "series"
                ),

                display_name=(
                    f"{model.display_name}"
                    " (Series)"
                ),

                cache=ApproximationCache(

                    baseline=model.baseline,

                    residual=model.residual,

                    coefficients=coeffs,

                    use_series=True,

                ),
            )
        )

    ####################################################
    # Exact SciPy
    ####################################################

    methods.append(

        ApproximationMethod(

            identifier="scipy",

            display_name="SciPy",

            exact=True,

        )
    )

    return methods


__all__ = [
    "ApproximationMethod",
    "build_methods",
]