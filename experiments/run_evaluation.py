"""
Run Phase 1 evaluation experiments.

Loads fitted coefficient sets produced by run_fitting.py.

Evaluates all models on:

    x in [1,2]

Produces:

results/

    gamma_metrics.csv
    gamma_metrics.json

    digamma_metrics.csv
    digamma_metrics.json

    trigamma_metrics.csv
    trigamma_metrics.json
"""

from __future__ import annotations


import json
from pathlib import Path

import numpy as np


from approximation.model import (
    ALL_APPROXIMATIONS,
)


from evaluation.evaluate import (
    run_evaluation,
)



# ============================================================
# Configuration
# ============================================================


RESULTS_DIR = Path(
    "results"
)


COEFFICIENT_FILE = (
    RESULTS_DIR
    /
    "fitted_coefficients.json"
)


EVALUATION_INTERVAL = (
    1.0,
    2.0,
)


NUM_POINTS = 1000



# ============================================================
# Model lookup
# ============================================================


def build_model_lookup():
    """
    Build mapping from serialized model ids
    to approximation objects.
    """

    return {
        model.identifier: model
        for model in ALL_APPROXIMATIONS
    }



# ============================================================
# Load fitted models
# ============================================================


def load_fitted_models(
    filename: Path,
):
    """
    Load fitted coefficient sets.

    Converts serialized JSON entries into
    evaluation-ready tuples:

        (
            model,
            coefficients,
            loss_type,
            loss_value,
        )
    """

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)


    model_lookup = build_model_lookup()


    fitted_models = []


    for result in results:

        model_id = result["model_id"]


        if model_id not in model_lookup:

            raise ValueError(
                f"Unknown model id: {model_id}"
            )


        model = model_lookup[model_id]


        coefficients = np.asarray(
            result["coefficients"],
            dtype=np.float64,
        )


        loss_type = result["loss_id"]

        loss_value = (
            result["objective_value"]
        )


        fitted_models.append(
            (
                model,
                coefficients,
                loss_type,
                loss_value,
            )
        )


    return fitted_models



# ============================================================
# Main experiment
# ============================================================


def main():

    print(
        "\nStarting Phase 1 evaluation..."
    )


    # --------------------------------------------------------
    # Evaluation grid
    # --------------------------------------------------------

    x = np.linspace(
        EVALUATION_INTERVAL[0],
        EVALUATION_INTERVAL[1],
        NUM_POINTS,
    )


    print(
        f"Interval: [{x.min()}, {x.max()}]"
    )

    print(
        f"Points: {len(x)}"
    )


    # --------------------------------------------------------
    # Load coefficients
    # --------------------------------------------------------

    fitted_models = load_fitted_models(
        COEFFICIENT_FILE
    )


    print(
        f"\nLoaded {len(fitted_models)} fitted models."
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    run_evaluation(
        fitted_models=fitted_models,
        x=x,
        output_dir=RESULTS_DIR,
    )


    print(
        "\nEvaluation complete."
    )

    print(
        f"Saved results to {RESULTS_DIR.resolve()}"
    )



if __name__ == "__main__":
    main()