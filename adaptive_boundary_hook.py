import numpy as np
import os

class AdaptiveBoundaryHook:
    def __init__(
        self,
        alpha=0.1,
        warmup_epsilon=0.01,       # epsilon iniziale durante il warm-up
        min_epsilon=0.01,          # minimo valore per ricerca
        max_epsilon=0.1,           # massimo valore per ricerca
        search_epsilon=True,
        log_path=None,
        verbose=False,
        warmup_epochs=7,
        max_delta_change=0.05,
        min_gap_change=0.001
    ):
        self.alpha = alpha
        self.warmup_epsilon = warmup_epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.search_epsilon = search_epsilon
        self.verbose = verbose
        self.log_path = log_path
        self.warmup_epochs = warmup_epochs
        self.max_delta_change = max_delta_change
        self.min_gap_change = min_gap_change
        self.delta = None

        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w') as f:
                f.write("epoch,epsilon,delta,gap\n")

    def find_adaptive_epsilon(self, ll):
        best_epsilon = self.min_epsilon
        best_gap = -np.inf
        for eps in np.linspace(self.min_epsilon, self.max_epsilon, 10):
            boundary = np.percentile(ll, eps * 100)
            inside = ll[ll >= boundary]
            outside = ll[ll < boundary]
            if len(inside) == 0 or len(outside) == 0:
                continue
            gap = np.min(inside) - np.max(outside)
            if gap > best_gap + self.min_gap_change:
                best_gap = gap
                best_epsilon = eps
        return best_epsilon, best_gap

    def update(self, ll, epoch=None):
        if epoch is not None and epoch < self.warmup_epochs:
            best_epsilon = self.warmup_epsilon
            best_gap = None
        else:
            best_epsilon, best_gap = self.find_adaptive_epsilon(ll) if self.search_epsilon else (self.warmup_epsilon, None)

        delta_estimate = np.percentile(ll, best_epsilon * 100)

        if self.delta is None:
            self.delta = delta_estimate
        else:
            delta_change = delta_estimate - self.delta
            delta_change = np.clip(delta_change, -self.max_delta_change, self.max_delta_change)
            self.delta += self.alpha * delta_change

        if self.verbose:
            print(
                f"[AdaptiveBoundaryHook] Epoch {epoch}: Δ={self.delta:.4f}, ε={best_epsilon:.4f}, Gap={best_gap:.4f}"
                if best_gap is not None else
                f"[AdaptiveBoundaryHook] Epoch {epoch}: Δ={self.delta:.4f} (warm-up)"
            )

        if self.log_path is not None and epoch is not None:
            gap_val = best_gap if best_gap is not None else -1
            with open(self.log_path, 'a') as f:
                f.write(f"{epoch},{best_epsilon:.4f},{self.delta:.4f},{gap_val:.4f}\n")

        return self.delta
