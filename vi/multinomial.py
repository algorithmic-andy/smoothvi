from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class MultinomialSample:

    alpha: np.ndarray

    theta: np.ndarray

    counts: np.ndarray

    posterior_lambda: np.ndarray


def generate_sample(
    alpha,
    total_count: int,
    rng: np.random.Generator,
):

    alpha = np.asarray(alpha, dtype=np.float64)

    theta = rng.dirichlet(alpha)

    counts = rng.multinomial(total_count, theta)

    posterior_lambda = alpha + counts

    return MultinomialSample(
        alpha=alpha,
        theta=theta,
        counts=counts.astype(np.float64),
        posterior_lambda=posterior_lambda.astype(np.float64),
    )