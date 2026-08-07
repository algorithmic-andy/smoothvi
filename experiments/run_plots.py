"""
Generate Phase 1 approximation figures.

Creates error plots for every fitted approximation and
saves them to

results/plots/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from approximation.model import ALL_APPROXIMATIONS

from evaluation.metrics import (
    gamma_target,
    digamma_target,
    trigamma_target,
)

from evaluation.plots import (
    plot_error_curve,
    plot_all_relative_errors,
    plot_factor_comparison,
    plot_max_relative_error,
    plot_correlation_heatmap,
)


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")

PLOTS_DIR = RESULTS_DIR / "plots"

COEFFICIENT_FILE = (
    RESULTS_DIR /
    "fitted_coefficients.json"
)

NUM_POINTS = 2000

INTERVAL = (
    1.0,
    2.0,
)


# ============================================================
# Helpers
# ============================================================


def build_lookup():

    return {
        model.identifier: model
        for model in ALL_APPROXIMATIONS
    }


def load_results():

    with open(
        COEFFICIENT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# Main
# ============================================================


def main():

    print(
        "\nGenerating Phase 1 plots...\n"
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lookup = build_lookup()

    buffer = 10

    x = np.linspace(
        INTERVAL[0],
        INTERVAL[1],
        NUM_POINTS,
    )[buffer:-buffer]

    gamma_true = gamma_target(x)
    digamma_true = digamma_target(x)
    trigamma_true = trigamma_target(x)

    fitted = load_results()

    summary_rows = []

    for result in fitted:

        model = lookup[
            result["model_id"]
        ]

        coefficients = np.asarray(
            result["coefficients"],
            dtype=np.float64,
        )

        loss = result["loss_id"]

        prefix = (
            f"{model.identifier}_{loss}"
        )

        # ----------------------------------------------------
        # Gamma
        # ----------------------------------------------------

        gamma_prediction = model(
            x,
            coefficients,
        )

        plot_error_curve(
            x,
            gamma_prediction,
            gamma_true,
            f"{prefix} Gamma",
            PLOTS_DIR /
            f"{prefix}_gamma.png",
        )
        
        # ----------------------------------------------------
        # Digamma
        # ----------------------------------------------------

        digamma_prediction = model.first(
            x,
            coefficients,
        )

        plot_error_curve(
            x,
            digamma_prediction,
            digamma_true,
            f"{prefix} Digamma",
            PLOTS_DIR /
            f"{prefix}_digamma.png",
        )
        
        # ----------------------------------------------------
        # Trigamma
        # ----------------------------------------------------

        trigamma_prediction = model.second(
            x,
            coefficients,
        )

        plot_error_curve(
            x,
            trigamma_prediction,
            trigamma_true,
            f"{prefix} Trigamma",
            PLOTS_DIR /
            f"{prefix}_trigamma.png",
        )
        
        summary_rows.append(
            {
                "loss": result["objective_value"],
                "gamma_rmse": np.sqrt(
                    np.mean(
                        (gamma_prediction - gamma_true) ** 2
                    )
                ),
                "digamma_rmse": np.sqrt(
                    np.mean(
                        (digamma_prediction - digamma_true) ** 2
                    )
                ),
                "trigamma_rmse": np.sqrt(
                    np.mean(
                        (trigamma_prediction - trigamma_true) ** 2
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    plot_correlation_heatmap(
        pd.DataFrame(summary_rows),
        PLOTS_DIR /
        "metric_correlation_heatmap.png",
    )

    print(
        f"Saved plots to:\n{PLOTS_DIR.resolve()}"
    )


if __name__ == "__main__":

    main()