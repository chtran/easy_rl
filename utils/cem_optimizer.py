import numpy as np


class CEMOptimizer:
    def __init__(self, fn, n_in, n_iters=2, n_samples=64, n_elites=6, distribution='Gaussian'):
        self.fn = fn
        self.n_in = n_in
        self.n_iters = n_iters
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.mean = np.zeros(self.n_in)
        self.var = np.diag(np.ones(self.n_in))

    def optimize(self):
        for i in range(self.n_iters):
            x = np.random.multivariate_normal(self.mean, self.var, size=self.n_samples)
            results = []
            for j in range(x.shape[0]):
                results.append((x[j,:], self.fn(x[j, :])))
            sorted_results = sorted(results, key=lambda tup: tup[1], reverse=True) 
            elites = [tup[0] for tup in sorted_results[:self.n_elites]]
            self.mean = np.mean(elites, axis=0)
            self.var = np.diag(np.var(elites, axis=0, ddof=1))
        return sorted_results[0]
