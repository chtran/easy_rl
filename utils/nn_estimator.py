import numpy as np
import tensorflow as tf

class NNEstimator(object):
    def __init__(self, 
            n_in, n_out, hidden_dims=[10, 10], learning_rate=3e-4, sess=None, scope=None):
        self.n_in = n_in
        self.n_out = n_out

        self._build_graph(hidden_dims, learning_rate, scope)
        if sess is None:
            self.sess = tf.Session()
            self.sess.__enter__()
        else:
            self.sess =  sess

    def _build_graph(self, hidden_dims, learning_rate, scope):
        with tf.variable_scope(scope):
            self.input_ph = tf.placeholder(shape=[None, self.n_in], name='input', dtype=tf.float32)
            hidden = self.input_ph
            for i, hidden_dim in enumerate(hidden_dims):
                hidden = tf.layers.dense(
                    hidden, hidden_dim, activation=tf.nn.relu, name=f'hidden_{i}')
            self.output = tf.layers.dense(hidden, self.n_out, name='output')
            self.labels_ph = tf.placeholder(shape=[None, self.n_out], name='labels', dtype=tf.float32)
            self.loss = tf.losses.mean_squared_error(self.labels_ph, self.output)
            self.fit_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def predict(self, x):
        return self.sess.run(self.output, feed_dict={self.input_ph: x})

    def fit(self, x, y):
        return self.sess.run(self.fit_op, feed_dict={self.input_ph: x, self.labels_ph: y})
