from improve2 import get_stable_boundary

class AdaptiveBoundaryHook:
    def __init__(self, alpha=0.2, use_dynamic_epsilon=True, epsilon=0.01, max_epsilon=0.05, n_bootstrap=100):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.n_bootstrap = n_bootstrap
        self.search_epsilon = use_dynamic_epsilon  # <--- questa riga è la modifica chiave
        self.delta = None  # inizializzato al primo batch
        


    def update(self, log_likelihoods):
        """
        Chiamata ad ogni batch per aggiornare dinamicamente il delta.
        log_likelihoods: array numpy dei log-likelihood normali (solo normali)
        """
        # Calcolo nuovo delta
        new_delta, used_epsilon = get_stable_boundary(
            log_likelihoods,
            epsilon=self.epsilon,
            max_epsilon=self.max_epsilon,
            n_bootstrap=self.n_bootstrap,
            search_epsilon=self.use_dynamic_epsilon
        )

        if self.delta is None:
            self.delta = new_delta
        else:
            self.delta = (1 - self.alpha) * self.delta + self.alpha * new_delta

        return self.delta, used_epsilon
