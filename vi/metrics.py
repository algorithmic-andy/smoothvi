"""
Metrics for Phase 2 experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_results(results: pd.DataFrame):

    grouped = results.groupby(
        [
            "approximation",
            "dimension",
            "variance",
        ]
    )

    summary = grouped.agg(

        mean_elbo_error=("elbo_error", "mean"),
        median_elbo_error=("elbo_error", "median"),
        sd_elbo_error=("elbo_error", "std"),

        mean_gamma_error=("gamma_error", "mean"),
        median_gamma_error=("gamma_error", "median"),
        sd_gamma_error=("gamma_error", "std"),

        mean_psi_error=("digamma_error", "mean"),
        median_psi_error=("digamma_error", "median"),
        sd_psi_error=("digamma_error", "std"),

        mean_tri_error=("trigamma_error", "mean"),
        median_tri_error=("trigamma_error", "median"),
        sd_tri_error=("trigamma_error", "std"),

    ).reset_index()

    summary["iqr_elbo"] = grouped["elbo_error"].quantile(0.75).values \
                        - grouped["elbo_error"].quantile(0.25).values
    
    summary["iqr_gamma"] = grouped["gamma_error"].quantile(0.75).values \
                            - grouped["gamma_error"].quantile(0.25).values
    
    summary["iqr_psi"] = grouped["digamma_error"].quantile(0.75).values \
                            - grouped["digamma_error"].quantile(0.25).values

    summary["iqr_tri"] = grouped["trigamma_error"].quantile(0.75).values \
                            - grouped["trigamma_error"].quantile(0.25).values

    summary["cv_elbo"] = (
        summary["sd_elbo_error"]
        /
        summary["mean_elbo_error"].abs().clip(lower=1e-12)
    )

    summary["cv_gamma"] = (
        summary["sd_gamma_error"]
        /
        summary["mean_gamma_error"].abs().clip(lower=1e-12)
    )

    summary["cv_psi"] = (
        summary["sd_psi_error"]
        /
        summary["mean_psi_error"].abs().clip(lower=1e-12)
    )

    summary["cv_tri"] = (
        summary["sd_tri_error"]
        /
        summary["mean_tri_error"].abs().clip(lower=1e-12)
    )

    summary["optimization_score"] = (
        summary["mean_gamma_error"]
        +
        summary["mean_psi_error"]
        +
        summary["mean_tri_error"]

    )

    summary["stability_score"] = (

        summary["mean_elbo_error"].abs()

        +

        summary["sd_elbo_error"]

    )

    return summary


def rank_models(summary):

    ranking = (

        summary

        .groupby("approximation")

        .agg(

            optimization=("optimization_score", "mean"),

            mean_tri_error=("mean_tri_error", "mean"),

            mean_psi_error=("mean_psi_error", "mean"),

            mean_gamma_error=("mean_gamma_error", "mean"),

            sd_tri_error=("sd_tri_error", "mean"),

            sd_psi_error=("sd_psi_error", "mean"),

            sd_gamma_error=("sd_gamma_error", "mean"),

            mean_elbo_error=("mean_elbo_error","mean"),

            sd_elbo=("sd_elbo_error","mean"),

            stability=("stability_score","mean"),

        )

        .reset_index()

    )

    ranking = ranking.sort_values(

        by=[

            "optimization",

            "mean_tri_error",

            "mean_psi_error",

            "mean_gamma_error",

            "sd_tri_error",

            "sd_psi_error",

            "sd_gamma_error",

            "stability",

        ],

        key=lambda x: np.abs(x),

    )

    ranking["rank"] = np.arange(

        1,

        len(ranking)+1,

    )

    return ranking


__all__ = [

    "summarize_results",

    "rank_models",

]