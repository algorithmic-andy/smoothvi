"""
Complete Phase 2 experiment.

Synthetic Dirichlet-Multinomial experiment.

For every regime:

1. Sample θ ~ Dirichlet(α)
2. Sample counts x ~ Multinomial(N, θ)
3. Compute the exact posterior

       λ* = α + x

4. Evaluate every approximation at λ*.

No numerical optimization is required because the
Dirichlet-Multinomial posterior is conjugate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .approximations import build_methods
from .regimes import build_regimes

from .elbo import (
    dirichlet_multinomial_elbo,
    elbo_gradient_error,
)

from .multinomial import (
    generate_sample,
)

from utils.config import (
    RANDOM_SEED,
    TOTAL_COUNT,
)


def run_experiment():

    rng = np.random.default_rng(RANDOM_SEED)

    methods = build_methods()

    scipy_method = next(

        method

        for method in methods

        if method.identifier == "scipy"

    )

    regimes = build_regimes()

    rows = []

    for regime in regimes:

        print(f"\n{regime.identifier}")

        for prior_sample in regime.samples:

            alpha = prior_sample.alpha

            ####################################################
            # Generate synthetic observations
            ####################################################

            observation = generate_sample(

                alpha=alpha,

                total_count=TOTAL_COUNT,

                rng=rng,

            )

            counts = observation.counts

            lambda_star = observation.posterior_lambda

            ####################################################
            # Exact SciPy quantities
            ####################################################

            reference_elbo = dirichlet_multinomial_elbo(

                alpha,

                counts,

                lambda_star,

                scipy_method,

            )

            ####################################################
            # Evaluate every approximation
            ####################################################

            for method in methods:

                elbo = dirichlet_multinomial_elbo(

                    alpha,

                    counts,

                    lambda_star,

                    method,

                )

                gam_error, psi_error, tri_error = elbo_gradient_error(
                    lambda_star,
                    method,
                    scipy_method,
                )

                rows.append(

                    {

                        "regime":
                            regime.identifier,

                        "replication":
                            prior_sample.replication,

                        "dimension":
                            regime.dimension,

                        "variance":
                            regime.variance,

                        "total_count":
                            TOTAL_COUNT,

                        "approximation":
                            method.identifier,

                        ################################################

                        "elbo":
                            float(elbo),

                        "elbo_error":
                            float(
                                elbo - reference_elbo
                            ),

                        ################################################

                        "gamma_error":
                            gam_error,

                        "digamma_error":
                            psi_error,

                        "trigamma_error":
                            tri_error,

                    }

                )

    return pd.DataFrame(rows)


__all__ = [
    "run_experiment",
]