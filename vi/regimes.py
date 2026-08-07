"""
Dirichlet experimental regimes.

Each regime is defined by

    • Dirichlet dimension
    • Dirichlet parameter variance

For each regime, multiple independent parameter vectors are generated
to provide replication in the simulation study.

Every approximation method is evaluated on exactly the same parameter
vectors, reducing Monte Carlo variability between treatments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from utils.config import (
    DIRICHLET_MEAN,
    DIRICHLET_FLOOR,
    LOW_DIMENSION,
    HIGH_DIMENSION,
    LOW_VARIANCE,
    HIGH_VARIANCE,
    DIRICHLET_REPLICATIONS,
    DIRICHLET_SEED,
)


@dataclass(frozen=True)
class DirichletSample:
    """
    One replicated Dirichlet parameter vector.
    """

    replication: int

    alpha: NDArray[np.float64]


@dataclass(frozen=True)
class DirichletRegime:
    """
    Experimental Dirichlet regime.
    """

    identifier: str

    dimension: int

    variance: float

    samples: list[DirichletSample]


def _generate_samples(
    dimension: int,
    variance: float,
    rng: np.random.Generator,
) -> list[DirichletSample]:
    """
    Generate replicated Dirichlet parameter vectors.
    """

    samples = []

    standard_deviation = np.sqrt(variance)

    for replication in range(DIRICHLET_REPLICATIONS):

        alpha = rng.normal(
            loc=DIRICHLET_MEAN,
            scale=standard_deviation,
            size=dimension,
        )

        alpha = np.maximum(
            alpha,
            DIRICHLET_FLOOR,
        )

        samples.append(
            DirichletSample(
                replication=replication,
                alpha=alpha.astype(np.float64),
            )
        )

    return samples


def build_regimes() -> list[DirichletRegime]:
    """
    Construct all experimental regimes.
    """

    rng = np.random.default_rng(
        DIRICHLET_SEED,
    )

    return [

        DirichletRegime(
            identifier="low5",
            dimension=LOW_DIMENSION,
            variance=LOW_VARIANCE,
            samples=_generate_samples(
                LOW_DIMENSION,
                LOW_VARIANCE,
                rng,
            ),
        ),

        DirichletRegime(
            identifier="low25",
            dimension=HIGH_DIMENSION,
            variance=LOW_VARIANCE,
            samples=_generate_samples(
                HIGH_DIMENSION,
                LOW_VARIANCE,
                rng,
            ),
        ),

        DirichletRegime(
            identifier="high5",
            dimension=LOW_DIMENSION,
            variance=HIGH_VARIANCE,
            samples=_generate_samples(
                LOW_DIMENSION,
                HIGH_VARIANCE,
                rng,
            ),
        ),

        DirichletRegime(
            identifier="high25",
            dimension=HIGH_DIMENSION,
            variance=HIGH_VARIANCE,
            samples=_generate_samples(
                HIGH_DIMENSION,
                HIGH_VARIANCE,
                rng,
            ),
        ),
    ]


__all__ = [
    "DirichletSample",
    "DirichletRegime",
    "build_regimes",
]