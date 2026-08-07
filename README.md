# Identity-Constrained Approximation Framework

*A general framework for constructing high-accuracy approximations of special functions using known functional identities.*

---

## Overview

This repository accompanies the Master's Research Paper

> **Identity-Constrained Approximation Framework for Special Functions**

The project introduces a general methodology for approximating special functions by incorporating known mathematical identities directly into the fitting process.

Rather than minimizing only pointwise approximation error, the framework constrains approximations to satisfy known structural identities such as recurrence relations, duplication formulas, and derivative identities. These constraints produce approximations that are not only accurate as functions, but also preserve important mathematical structure that benefits downstream computation.

The Gamma function is used as the primary case study because it possesses a rich collection of functional identities together with an important role in Bayesian statistics.

---

# Motivation

Most approximation methods optimize only function values.

Many statistical algorithms repeatedly differentiate these approximations, causing small pointwise errors to accumulate into much larger optimization errors.

This framework instead asks:

> **Can incorporating mathematical identities during fitting improve downstream statistical performance?**

To answer this question, we construct several approximation families for the logarithm of the Gamma function and evaluate them both mathematically and statistically.

---

# Framework

The framework consists of two stages.

```
Mathematical identities
        │
        ▼
Construct approximation family
        │
        ▼
Optimize coefficients
        │
        ▼
Freeze approximation
        │
        ▼
Evaluate in Bayesian inference
```

The methodology itself is general and is not restricted to the Gamma function.

---

# Repository Structure

```
approximation/
    Baselines
    Residual models
    Approximation construction

evaluation/
    Benchmarking

experiments/
    Phase 1 experiments

fitting/
    Losses
    Coefficient optimization

vi/
    Dirichlet–Multinomial experiments
    Bayesian inference evaluation
    ELBO comparison

scripts/
  Run Phase 1 only
  Run Phase 2 only
  Run Phase 1 + Phase 2

results/
    Optimized coefficients
    Experimental summaries

plots/
    Thesis figures

tests/
    Unit tests

utils/
    Shared utilities
```

---

# Approximation Models

Each approximation consists of

```
Approximation

    = Baseline

    + Residual
```

Two baselines are considered

- Polynomial
- Stirling

Two residual families are considered

- Bernstein
- Rational

Three fitting objectives are considered

- Least Squares
- Identity-Constrained
- Duplication-Constrained

Each approximation can then obtain derivatives using either

- Direct differentiation
- Independent derivative approximation

producing twelve final approximation methods.

---

# Phase 1

Phase 1 constructs approximations to

$$
\log \Gamma(x)
$$

on

$$
x\in[1,2].
$$

Approximations are evaluated using

- Log-Gamma error
- Digamma error
- Trigamma error
- Approximation score
- Functional identity violations

The optimized coefficients are frozen before Phase 2.

---

# Phase 2

Phase 2 evaluates the approximations inside Bayesian inference.

A synthetic Dirichlet–Multinomial model is generated.

For each replication

1. Sample

$$
\theta\sim\text{Dirichlet}(\alpha)
$$

2. Sample observations

$$
x\sim\text{Multinomial}(N,\theta)
$$

3. Compute the exact posterior

$$
\lambda^\*= \alpha+x
$$

4. Evaluate every approximation at the same posterior.

Each approximation is compared using

- ELBO error
- Digamma error
- Trigamma error
- Approximation score
- Stability across replications

This isolates the effect of the approximation itself.

---

# Installation

Clone the repository

```bash
git clone https://github.com/algorithmic-andy/smoothvi.git
```

Install

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

# Running Phase 1

```bash
python run_phase1.py
```

Outputs include

- fitted coefficients
- approximation summaries
- figures

---

# Running Phase 2

```bash
python run_phase2.py
```

Outputs include

- ELBO comparisons
- Bayesian benchmark results
- model rankings

---

# Main Contributions

This work introduces

- Identity-constrained approximation as a general optimization framework.
- A modular decomposition into baselines and residual models.
- Statistical evaluation of approximation quality inside Bayesian inference.
- Evidence that preserving mathematical identities can improve downstream statistical performance despite slightly larger pointwise approximation error.

---

# Future Work

This framework naturally extends to

- Continuous transformations of the Gamma function
- Beta function
- Multivariate Gamma function
- Barnes G-function
- Hurwitz Zeta function
- Completed Riemann Zeta function
- Other special functions possessing structural identities

The framework is intended to provide a reusable methodology rather than a Gamma-specific approximation.

---

# Citation

If you use this repository, please cite

```
Joshua King (2026)

Identity-Constrained Approximation Framework for Special Functions

Master's Research Paper
University of Waterloo
```

<img width="191" height="20" alt="image" src="https://github.com/user-attachments/assets/51433965-0975-4b1e-ba9f-f1a677e815fe" />


---

# License

This project is released under the MIT License.

---

# Acknowledgements

This work was completed as part of a Master's Research Paper in Statistics and benefited from discussions with supervisor Paul Marriott.

The Gamma function serves as the motivating case study, while the proposed methodology is intended as a general framework for identity-constrained approximation of special functions.
