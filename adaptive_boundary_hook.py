import numpy as np
import os

class AdaptiveBoundaryHook:
    def __init__(
        self,
        alpha=0.2,
        epsilon=0.01,
        max_epsilon=0.05,
        n_bootstrap=100,
        search_epsilon=True,
        log_path=None,
        verbose=True
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.n_bootstrap = n_bootstrap
        self.search_epsilon = search_epsilon
        self.verbose = verbose
        self.delta = None
        self.log_path = log_path

        if self.log_path is not None:
            with open(self.log_path, 'w') as f:
                f.write("epoch,epsilon,delta,std_delta\n")

    def find_adaptive_epsilon(self, ll):## ll è la patch 
        best_epsilon = self.epsilon
        best_gap = -np.inf

        for eps in np.linspace(self.epsilon, self.max_epsilon, 10):
            boundary = np.percentile(ll, eps * 100)
            inside = ll[ll >= boundary]
            outside = ll[ll < boundary]
            if len(inside) == 0 or len(outside) == 0:
                continue
            gap = np.min(inside) - np.max(outside)
            if gap > best_gap:
                best_gap = gap
                best_epsilon = eps
        return best_epsilon

    def bayesian_boundary_estimate(self, ll, epsilon):
        n = len(ll)
        boundaries = []#saranno contenuti i confini dei diversi bootstrap
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(ll, n, replace=True)
            b = np.percentile(sample, epsilon * 100)
            boundaries.append(b)
        return np.mean(boundaries), np.std(boundaries)

    def update(self, ll, epoch=None):
        if self.search_epsilon:
            best_epsilon = self.find_adaptive_epsilon(ll)
        else:
            best_epsilon = self.epsilon

        delta_mean, delta_std = self.bayesian_boundary_estimate(ll, best_epsilon)

        if self.delta is None:
            self.delta = delta_mean
        else:
            self.delta = (1 - self.alpha) * self.delta + self.alpha * delta_mean

        if self.verbose:
            print(f"[AdaptiveBoundaryHook] Epoch {epoch}: Δ={self.delta:.4f}, ε={best_epsilon:.4f}, ±{delta_std:.4f}")

        if self.log_path is not None and epoch is not None:
            with open(self.log_path, 'a') as f:
                f.write(f"{epoch},{best_epsilon:.4f},{self.delta:.4f},{delta_std:.4f}\n")

        return self.delta
