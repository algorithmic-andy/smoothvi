"""
Run the complete Phase 2 variational inference study.

Pipeline
--------

1. Load frozen approximation methods
2. Generate replicated Dirichlet regimes
3. Compute one SciPy optimum per replicated regime
4. Evaluate every approximation at the common optimum
5. Save raw results
6. Compute summary statistics
7. Rank approximation methods
8. Produce plots
"""

from __future__ import annotations

import json
from pathlib import Path

from vi.experiment import run_experiment

from vi.metrics import (
    summarize_results,
    rank_models,
)

from vi.plots import (
    plot_gamma,
    plot_digamma,
    plot_trigamma,
    plot_ranking,
    plot_stability,
    plot_top10,
)


RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Running Phase 2 experiment...")

    ####################################################
    # Experiment
    ####################################################

    results = run_experiment()

    ####################################################
    # Save raw results
    ####################################################

    results.to_csv(
        RESULTS_DIR / "vi_results.csv",
        index=False,
    )

    with open(
        RESULTS_DIR / "vi_results.json",
        "w",
    ) as f:

        json.dump(
            results.to_dict(
                orient="records",
            ),
            f,
            indent=4,
        )

    ####################################################
    # Metrics
    ####################################################

    summary = summarize_results(
        results,
    )

    rankings = rank_models(
        summary,
    )

    summary.to_csv(
        RESULTS_DIR / "vi_summary.csv",
        index=False,
    )

    rankings.to_csv(
        RESULTS_DIR / "vi_rankings.csv",
        index=False,
    )

    with open(
        RESULTS_DIR / "vi_summary.json",
        "w",
    ) as f:

        json.dump(
            summary.to_dict(
                orient="records",
            ),
            f,
            indent=4,
        )

    with open(
        RESULTS_DIR / "vi_rankings.json",
        "w",
    ) as f:

        json.dump(
            rankings.to_dict(
                orient="records",
            ),
            f,
            indent=4,
        )

    ####################################################
    # Figures
    ####################################################

    plot_ranking(
        rankings,
        PLOTS_DIR / "model_ranking.png",
    )

    plot_gamma(
        rankings,
        PLOTS_DIR / "model_gamma.png",
    )

    plot_digamma(
        rankings,
        PLOTS_DIR / "model_digamma.png",
    )

    plot_trigamma(
        rankings,
        PLOTS_DIR / "model_trigamma.png",
    )

    plot_stability(
        rankings,
        PLOTS_DIR / "model_stability.png",
    )

    plot_top10(
        rankings,
        PLOTS_DIR / "model_top10.png",
    )

    print()

    print("====================================")
    print(" Phase 2 Complete")
    print("====================================")

    print(f"Results : {RESULTS_DIR}")
    print(f"Plots   : {PLOTS_DIR}")


if __name__ == "__main__":

    main()