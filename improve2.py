import numpy as np

def find_adaptive_epsilon(normal_log_likelihoods, start_epsilon=0.01, max_epsilon=0.05, steps=10):
    """
    Trova il miglior epsilon adattivo in base al massimo gap tra inside e outside likelihoods.
    """
    best_epsilon = start_epsilon
    best_gap = -np.inf

    for epsilon in np.linspace(start_epsilon, max_epsilon, steps):
        boundary = np.percentile(normal_log_likelihoods, epsilon * 100)
        inside = normal_log_likelihoods[normal_log_likelihoods >= boundary]
        outside = normal_log_likelihoods[normal_log_likelihoods < boundary]

        if len(inside) == 0 or len(outside) == 0:
            continue

        gap = np.min(inside) - np.max(outside)
        if gap > best_gap:
            best_gap = gap
            best_epsilon = epsilon

    return best_epsilon

def bayesian_boundary_estimate(normal_log_likelihoods, epsilon=0.01, n_bootstrap=100):
    """
    Stima robusta di delta tramite bootstrap.
    """
    n_samples = len(normal_log_likelihoods)
    boundaries = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(normal_log_likelihoods, n_samples, replace=True)
        boundary = np.percentile(sample, epsilon * 100)
        boundaries.append(boundary)

    mean_boundary = np.mean(boundaries)
    std_boundary = np.std(boundaries)

    return mean_boundary, std_boundary

def get_stable_boundary(normal_log_likelihoods, start_epsilon=0.01, max_epsilon=0.05, n_bootstrap=100):
    """
    Calcola delta stabile combinando la ricerca dell'epsilon ottimo con la stima bayesiana.
    """
    best_epsilon = find_adaptive_epsilon(normal_log_likelihoods, start_epsilon, max_epsilon)
    delta, delta_std = bayesian_boundary_estimate(normal_log_likelihoods, epsilon=best_epsilon, n_bootstrap=n_bootstrap)
    return delta, delta_std, best_epsilon
