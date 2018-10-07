import numpy as np

class LinearEstimator(object):
    def __init__(self, 
            n_in, n_out, 
            learning_rate=0.01, regularization=0.001, add_bias=False,
            W=None):
        if W is not None:
            self.W = np.copy(W)
        elif add_bias:
            self.W = np.random.randn((n_in + 1), n_out) * 0.1
        else:
            self.W = np.random.randn(n_in, n_out) * 0.1
        self.n_in = n_in
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.add_bias = add_bias

    def predict(self, x, add_noise=False):
        """
        x: n_in 
        x_with_bias: n_in (+ 1) 
        W: n_in +1 x n_out
        Return: n_out
        """
        x = np.asarray(x)
        if self.add_bias:
            x = np.append(np.asarray(x), 1)

        result = self.W.T.dot(x)
        noise = 0.1 * np.random.randn(self.n_out) if add_noise else 0
        return result + noise

    def partial_grad(self, start_index, end_index):
        return self.W[start_index:end_index, :]

    def fit(self, x, y):
        """
        x: |S|
        y: |A|
        Minimize |f(x) - y| ^ 2
        """
        predicted_value = self.predict(x) # |A|
        delta = predicted_value - y #  |A|
        self.fit_to_delta(x, delta)

    def fit_to_delta(self, x, delta):
        """
        x: |S|
        delta: 1
        """
        x = np.asarray(x)
        if self.add_bias:
            x = np.append(np.asarray(x), 1)
        grad = np.outer(x, delta) + self.W * self.regularization
        self.W -= self.learning_rate *  grad
