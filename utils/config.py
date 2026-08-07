"""
Experiment configuration.

Contains numerical settings controlling approximation fitting
and experiments.
"""

from __future__ import annotations


# ============================================================
# Precomputation
# ============================================================

# Number of points used to discretize [1,2].
#
# Used for:
# - least squares fitting
# - identity loss evaluation
# - cached approximation evaluation
#
# Larger values improve integral approximation accuracy but
# increase optimization cost.
PRECOMPUTE_GRID_SIZE = 1000

EVAL_GRID_SIZE = 1000


# ============================================================
# Fitting Loss Weights
# ============================================================

IDENTITY_WEIGHTS = (10.0, 5.0, 3.0, 1.0)
LEAST_SQUARE_WEIGHTS = (5.0, 3.0, 1.0)


# ============================================================
# Differential evolution settings
# ============================================================

DE_SEED = 42

DE_POPSIZE = 75

DE_MAXITER = 1500

DE_TOL = 1e-12

DE_POLISH = True


# ==========================================================
# Dirichlet experiment settings
# ==========================================================

DIRICHLET_MEAN = 0.15

DIRICHLET_FLOOR = 1e-6

LOW_DIMENSION = 25
HIGH_DIMENSION = 100

LOW_VARIANCE = 0.10
HIGH_VARIANCE = 0.25

DIRICHLET_REPLICATIONS = 1000

TOTAL_COUNT = 10000
RANDOM_SEED = 42

DIRICHLET_SEED = 42

LAMBDA_FLOOR = 1e-8

INITIALIZATION_NOISE = 0.05


# ==========================================================
# Beta test settings
# ==========================================================

BETA_BOUNDS = (0.1, 10.0)
BETA_REPLICATIONS = 10