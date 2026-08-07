from __future__ import annotations

import numpy as np


def dirichlet_multinomial_elbo(
    alpha,
    counts,
    variational_lambda,
    method,
):

    alpha = np.asarray(alpha, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    variational_lambda = np.asarray(variational_lambda, dtype=np.float64)

    alpha0 = np.sum(alpha)
    lambda0 = np.sum(variational_lambda)

    lg_lambda = method.log_gamma(variational_lambda)
    lg_sum = method.log_gamma(lambda0)

    lg_alpha = method.log_gamma(alpha)
    lg_alpha_sum = method.log_gamma(alpha0)

    psi_lambda = method.digamma(variational_lambda)
    psi_sum = method.digamma(lambda0)

    expected_log_theta = psi_lambda - psi_sum

    prior_term = (
        lg_alpha_sum
        - np.sum(lg_alpha)
        + np.sum((alpha - 1.0) * expected_log_theta)
    )

    likelihood_term = np.sum(
        counts * expected_log_theta
    )

    entropy_term = (
        lg_sum
        - np.sum(lg_lambda)
        + np.sum((variational_lambda - 1.0) * expected_log_theta)
    )

    return prior_term + likelihood_term - entropy_term


def elbo_gradient_error(
    variational_lambda,
    method,
    scipy_method,
):
    
    gam_approx = method.log_gamma(variational_lambda)
    gam_exact = scipy_method.log_gamma(variational_lambda)
    gam_error = float(np.sqrt(np.mean((gam_approx - gam_exact) ** 2)))

    psi_approx = method.digamma(variational_lambda)
    psi_exact = scipy_method.digamma(variational_lambda)
    psi_error = float(np.sqrt(np.mean((psi_approx - psi_exact) ** 2)))

    tri_approx = method.trigamma(variational_lambda)
    tri_exact = scipy_method.trigamma(variational_lambda)
    tri_error = float(np.sqrt(np.mean((tri_approx - tri_exact) ** 2)))

    return gam_error, psi_error, tri_error