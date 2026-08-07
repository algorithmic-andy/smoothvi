"""
Run evaluation experiments for fitted log-Gamma approximations.

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


from pathlib import Path

import json

import numpy as np
import pandas as pd


from approximation.model import ALL_APPROXIMATIONS

from utils.routing import (
    route_to_core_interval,
    route_digamma_to_core_interval,
    route_trigamma_to_core_interval,
)


from .metrics import (
    rmse,
    sd,
    max_error,
    pearson_correlation,
    duplication_rmse,
    convexity_violation_rate,
    approximation_score,
    gamma_target,
    digamma_target,
    trigamma_target,
)


# ==============================================================
# Routed evaluation functions
# ==============================================================


def evaluate_loggamma(
    model,
    coefficients,
    x,
):
    """
    Evaluate routed log-Gamma approximation.

    Returns:

        sign * approximation(routed_x)
        + correction
    """

    routed = route_to_core_interval(x)

    return (
        routed.sign
        *
        model(
            routed.x,
            coefficients,
        )
        +
        routed.correction
    )


def evaluate_digamma(
    model,
    coefficients,
    x,
):
    """
    Evaluate routed Digamma approximation.
    """

    routed = route_digamma_to_core_interval(
        x
    )

    return (
        model.first(
            routed.x,
            coefficients,
        )
        +
        routed.correction
    )


def evaluate_trigamma(
    model,
    coefficients,
    x,
):
    """
    Evaluate routed Trigamma approximation.
    """

    routed = route_trigamma_to_core_interval(
        x
    )

    return (
        model.second(
            routed.x,
            coefficients,
        )
        +
        routed.correction
    )


# ==============================================================
# Model evaluation
# ==============================================================


def evaluate_model(
    model,
    coefficients,
    x,
    loss_name,
    loss_value,
):
    """
    Evaluate a single fitted model.
    """


    gamma_prediction = evaluate_loggamma(
        model,
        coefficients,
        x,
    )

    digamma_prediction = evaluate_digamma(
        model,
        coefficients,
        x,
    )

    trigamma_prediction = evaluate_trigamma(
        model,
        coefficients,
        x,
    )


    gamma_true = gamma_target(x)
    digamma_true = digamma_target(x)
    trigamma_true = trigamma_target(x)


    gamma_row = {

        "model": model.identifier,
        "loss_type": loss_name,
        "loss": loss_value,

        "rmse": rmse(
            gamma_prediction,
            gamma_true,
        ),

        "sd": sd(
            gamma_prediction,
            gamma_true,
        ),

        "max_error": max_error(
            gamma_prediction,
            gamma_true,
        ),

        "pearson": pearson_correlation(
            gamma_prediction,
            gamma_true,
        ),

        "duplication_rmse": duplication_rmse(
            model,
            coefficients,
            x,
        ),
    }


    digamma_row = {

        "model": model.identifier,
        "loss_type": loss_name,
        "loss": loss_value,

        "rmse": rmse(
            digamma_prediction,
            digamma_true,
        ),

        "sd": sd(
            digamma_prediction,
            digamma_true,
        ),

        "max_error": max_error(
            digamma_prediction,
            digamma_true,
        ),

        "pearson": pearson_correlation(
            digamma_prediction,
            digamma_true,
        ),

        "approximation": approximation_score(
            gamma_prediction,
            gamma_true,
            digamma_prediction,
            digamma_true,
            trigamma_prediction,
            trigamma_true,
        )
    }


    trigamma_row = {

        "model": model.identifier,
        "loss_type": loss_name,
        "loss": loss_value,

        "rmse": rmse(
            trigamma_prediction,
            trigamma_true,
        ),

        "sd": sd(
            trigamma_prediction,
            trigamma_true,
        ),

        "max_error": max_error(
            trigamma_prediction,
            trigamma_true,
        ),

        "pearson": pearson_correlation(
            trigamma_prediction,
            trigamma_true,
        ),

        "convexity_violation_rate":
            convexity_violation_rate(
                trigamma_prediction
            ),
    }


    return (
        gamma_row,
        digamma_row,
        trigamma_row,
    )


# ==============================================================
# Saving
# ==============================================================


def save_results(
    rows,
    filename,
    output_dir,
):
    """
    Save dataframe as CSV and JSON.
    """

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_dir / f"{filename}.csv",
        index=False,
    )


    with open(
        output_dir / f"{filename}.json",
        "w",
    ) as f:

        json.dump(
            rows,
            f,
            indent=4,
        )


# ==============================================================
# Main evaluation
# ==============================================================


def run_evaluation(
    fitted_models,
    x,
    output_dir="results",
):

    gamma_rows = []
    digamma_rows = []
    trigamma_rows = []


    for entry in fitted_models:

        (
            model,
            coefficients,
            loss_name,
            loss_value,
        ) = entry


        (
            gamma_row,
            digamma_row,
            trigamma_row,

        ) = evaluate_model(
            model,
            coefficients,
            x,
            loss_name,
            loss_value,
        )


        gamma_rows.append(gamma_row)
        digamma_rows.append(digamma_row)
        trigamma_rows.append(trigamma_row)



    output_dir = Path(
        output_dir
    )


    save_results(
        gamma_rows,
        "gamma_metrics",
        output_dir,
    )

    save_results(
        digamma_rows,
        "digamma_metrics",
        output_dir,
    )

    save_results(
        trigamma_rows,
        "trigamma_metrics",
        output_dir,
    )


__all__ = [
    "run_evaluation",
    "evaluate_loggamma",
    "evaluate_digamma",
    "evaluate_trigamma",
]