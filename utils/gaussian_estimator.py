import numpy as np

class GaussianEstimator:
    def __init__(self, n_in, n_out, add_bias=False):
        if add_bias:
            self.W =  np.random.randn((n_in + 1), n_out) * 0.1
        else:
            self.W = np.random.randn(n_in, n_out) * 0.1
        self.n_in  = n_in
        self.n_out = n_out
        self.add_bias = add_bias

    def get_sample(self, x):
        if self.add_bias:
            x = np.append(np.asarray(x), 1)
        mean = self.W.T.dot(x)
        return np.random.multivariate_normal(mean, np.diag(np.ones(self.n_out)))

    def get_log_prob(self, x, y):
        if self.add_bias:
            x = np.append(np.asarray(x), 1)
        mean = self.W.T.dot(x) # n_out * 1
        delta = y - mean # n_out * 1
        return -0.5 * self.n_out * np.log(2*np.pi) - 0.5 * delta.dot(delta)

    def get_grad_log_prob(self, x ,y):
        if self.add_bias:
            x = np.append(np.asarray(x), 1)
        mean = self.W.T.dot(x)
        delta = y - mean
        return np.outer(x, delta)

    def fit(self, x, y, value):
        self.W -= self.get_grad_log_prob(x, y) * value
