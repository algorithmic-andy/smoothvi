"""
Run coefficient fitting experiments.

This script fits all approximation models under all fitting objectives.

Experiment factors
------------------

Approximation structure:

    - Polynomial + Bernstein
    - Polynomial + Rational
    - Stirling + Bernstein
    - Stirling + Rational


Fitting objectives:

    - Least squares
    - Duplication constrained
    - Identity constrained


All Differential Evolution runs share the same initial population
to reduce optimization randomness during comparison.
"""

from __future__ import annotations


import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


from approximation.model import (
    ALL_APPROXIMATIONS,
)


from fitting.fit import (
    FitResult,
)


from fitting.fit import (
    fit_all_models,
)


from approximation.constants import (
    NUM_COEFFICIENTS,
)



# ============================================================
# Output configuration
# ============================================================


RESULTS_DIR = Path(
    "results"
)


COEFFICIENT_FILE = (
    RESULTS_DIR
    /
    "fitted_coefficients.json"
)



# ============================================================
# Serialization helper
# ============================================================


def _serialize_result(
    result: FitResult,
) -> dict:
    """
    Convert FitResult into JSON-compatible dictionary.
    """

    return {
        "model_id": result.model_id,

        "loss_id": result.loss_id,

        "coefficients": (
            result.coefficients
            .tolist()
        ),

        "objective_value": (
            result.objective_value
        ),

        "success": (
            result.success
        ),

        "message": (
            result.message
        ),
    }



# ============================================================
# Main experiment
# ============================================================


def main() -> None:
    """
    Execute full Phase 1 fitting experiment.
    """

    print(
        "Starting log-Gamma coefficient fitting..."
    )


    RESULTS_DIR.mkdir(
        exist_ok=True
    )


    results = fit_all_models()


    print(
        f"Completed {len(results)} fits."
    )


    serialized = [
        _serialize_result(result)
        for result in results
    ]


    with open(
        COEFFICIENT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serialized,
            file,
            indent=4,
        )


    print(
        f"Saved results to {COEFFICIENT_FILE}"
    )


    print(
        "\nSummary:"
    )


    for result in results:

        print(
            f"{result.model_id:35s}"
            f"{result.loss_id:20s}"
            f"loss={result.objective_value:.6e}"
        )



if __name__ == "__main__":

    main()