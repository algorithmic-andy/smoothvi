"""
Publication-quality plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare(path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def plot_ranking(ranking, output):

    _prepare(output)

    ranking = ranking.sort_values(
        "rank"
    )

    plt.figure(figsize=(12,10))

    plt.barh(

        ranking["approximation"],

        np.abs(ranking["mean_elbo_error"]),

    )

    plt.xscale("log")

    plt.xlabel("|Mean ELBO Error|")

    plt.ylabel("Approximation")

    plt.title("Overall ELBO Accuracy")

    plt.tight_layout()

    plt.savefig(output,dpi=300)

    plt.close()


def plot_gamma(ranking, output):

    _prepare(output)

    ranking = ranking.sort_values(

        "mean_gamma_error"

    )

    plt.figure(figsize=(12,10))

    plt.barh(

        ranking["approximation"],

        ranking["mean_gamma_error"],

    )

    plt.xscale("log")

    plt.xlabel("Gamma Error")

    plt.tight_layout()

    plt.savefig(output,dpi=300)

    plt.close()


def plot_digamma(ranking, output):

    _prepare(output)

    ranking = ranking.sort_values(

        "mean_psi_error"

    )

    plt.figure(figsize=(12,10))

    plt.barh(

        ranking["approximation"],

        ranking["mean_psi_error"],

    )

    plt.xscale("log")

    plt.xlabel("Digamma Error")

    plt.tight_layout()

    plt.savefig(output,dpi=300)

    plt.close()


def plot_trigamma(ranking, output):

    _prepare(output)

    ranking = ranking.sort_values(

        "mean_tri_error"

    )

    plt.figure(figsize=(12,10))

    plt.barh(

        ranking["approximation"],

        ranking["mean_tri_error"],

    )

    plt.xscale("log")

    plt.xlabel("Trigamma Error")

    plt.tight_layout()

    plt.savefig(output,dpi=300)

    plt.close()


def plot_stability(ranking, output):

    _prepare(output)

    ranking = ranking.sort_values(

        "stability"

    )

    plt.figure(figsize=(12,10))

    plt.barh(

        ranking["approximation"],

        ranking["stability"],

    )

    plt.xscale("log")

    plt.xlabel("Stability Score")

    plt.tight_layout()

    plt.savefig(output,dpi=300)

    plt.close()


def plot_top10(ranking, output):

    _prepare(output)

    top = ranking.nsmallest(

        10,

        "rank",

    )

    plt.figure(figsize=(10,5))

    plt.bar(

        top["approximation"],

        np.abs(top["mean_elbo_error"]),

    )

    plt.xticks(rotation=45,ha="right")

    plt.yscale("log")

    plt.ylabel("|Mean ELBO Error|")

    plt.tight_layout()

    plt.savefig(output,dpi=300)

    plt.close()


__all__ = [

    "plot_ranking",

    "plot_gradient",

    "plot_stability",

    "plot_top10",

]