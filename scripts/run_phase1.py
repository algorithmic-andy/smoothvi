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

from experiments.run_fitting import main as fit_main
from experiments.run_evaluation import main as eval_main
from experiments.run_plots import main as plot_main



# ============================================================
# Main
# ============================================================


def main():

    print(
        "\nGenerating Phase 1 coefficients...\n"
    )

    fit_main

    print(
        "\nGenerating Phase 1 evaluation...\n"
    )

    eval_main()

    plot_main()

    print(
        "\nPhase 1 Complete!\n"
    )
    


if __name__ == "__main__":

    main()