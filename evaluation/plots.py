"""
Plotting utilities for approximation evaluation.

Produces publication-quality comparison figures
rather than one plot per approximation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


EPSILON = 1e-12


# ==========================================================
# Utilities
# ==========================================================

def _prepare(path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _relative_error(
    prediction,
    target,
):

    return (

        np.abs(prediction - target)

        /

        (np.abs(target) + EPSILON)

    )


# ==========================================================
# Individual diagnostic plot
# ==========================================================

def plot_error_curve(
    x,
    prediction,
    target,
    label,
    output_path,
):
    """
    Individual diagnostic plot.

    Mainly useful while debugging.
    """

    absolute = np.abs(
        prediction - target
    )

    relative = _relative_error(
        prediction,
        target,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    axes[0].plot(
        x,
        absolute,
        linewidth=2,
    )

    axes[0].set_yscale("log")

    axes[0].set_title(
        "Absolute Error"
    )

    axes[0].set_xlabel("x")

    axes[0].set_ylabel("|Error|")

    axes[1].plot(
        x,
        relative,
        linewidth=2,
    )

    axes[1].set_yscale("log")

    axes[1].set_title(
        "Relative Error"
    )

    axes[1].set_xlabel("x")

    axes[1].set_ylabel(
        "Relative Error"
    )

    fig.suptitle(label)

    plt.tight_layout()

    _prepare(output_path)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# Compare every approximation
# ==========================================================

def plot_all_relative_errors(
    x,
    predictions,
    target,
    output_path,
):
    """
    Plot every approximation together.

    predictions:

        {
            label : prediction
        }
    """

    plt.figure(
        figsize=(12, 7)
    )

    for prediction in predictions:

        plt.plot(

            x,

            _relative_error(
                prediction,
                target,
            ),

            linewidth=2

        )

    plt.yscale("log")

    plt.xlabel("x")

    plt.ylabel("Relative Error")

    plt.title(
        "Relative Error Comparison"
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    _prepare(output_path)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# Compare one experimental factor
# ==========================================================

def plot_factor_comparison(
    x,
    predictions,
    target,
    factor,
    output_path,
):
    """
    Compare a single design factor.

    factor examples

        "Polynomial"

        "Stirling"

        "Bernstein"

        "Rational"

        "Least Squares"

        "Identity"

        "Duplication"
    """

    plt.figure(
        figsize=(10, 6)
    )

    for label, prediction in predictions.items():

        if factor.lower() not in label.lower():
            continue

        plt.plot(

            x,

            _relative_error(
                prediction,
                target,
            ),

            linewidth=2,

            label=label,

        )

    plt.yscale("log")

    plt.xlabel("x")

    plt.ylabel("Relative Error")

    plt.title(
        factor
    )

    plt.legend()

    plt.tight_layout()

    _prepare(output_path)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# Worst-case ranking
# ==========================================================

def plot_max_relative_error(
    predictions,
    target,
    output_path,
):
    """
    Rank methods by maximum relative error.
    """

    labels = []

    values = []

    for prediction in predictions:

        values.append(

            np.max(

                _relative_error(
                    prediction,
                    target,
                )

            )

        )

    order = np.argsort(values)

    #labels = np.array(labels)[order]

    values = np.array(values)[order]

    plt.figure(
        figsize=(10, 6)
    )

    #plt.barh(
     #   labels,
      #  values,
    #)

    plt.xscale("log")

    plt.xlabel(
        "Maximum Relative Error"
    )

    plt.title(
        "Worst-Case Relative Error"
    )

    plt.tight_layout()

    _prepare(output_path)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# Correlation Heatmap
# ==========================================================

def plot_correlation_heatmap(
    dataframe,
    output_path,
):

    import seaborn as sns

    corr = dataframe.corr(
        numeric_only=True,
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(

        corr,

        annot=True,

        cmap="coolwarm",

        center=0,

    )

    plt.tight_layout()

    _prepare(output_path)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


__all__ = [

    "plot_error_curve",

    "plot_all_relative_errors",

    "plot_factor_comparison",

    "plot_max_relative_error",

    "plot_correlation_heatmap",

]