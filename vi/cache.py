"""
Cache object for one frozen approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from approximation.baselines import Baseline
from approximation.residuals import Residual


@dataclass(frozen=True)
class ApproximationCache:

    baseline: Baseline

    residual: Residual

    coefficients: np.ndarray

    use_series: bool = False