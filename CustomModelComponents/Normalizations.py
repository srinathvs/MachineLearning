import tensorflow
import tensorflow as tf
import numpy as np


class SpectralNorm(tf.keras.layers.Layer):
    def __init__(self, n_iter=5, **kwargs):
        self.n_iter = n_iter
        super(SpectralNorm, self).__init__(**kwargs)

    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))

        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)

            u = tf.matmul(w, v)
            u /= tf.norm(u)

        spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a=True)

        return input_weights / spec_norm


class ConditionBatchNorm(tf.keras.layers.Layer):
    def __init__(self, n_class=2, decay_rate=0.999, eps=1e-7):
        super(ConditionBatchNorm, self).__init__()
        self.n_class = n_class
        self.decay = decay_rate
        self.eps = 1e-5

    def build(self, input_shape):
        self.input_size = input_shape
        n, h, w, c = input_shape

        self.gamma = self.add_weight(shape=[self.n_class, c],
                                     initializer='ones',
                                     trainable=True, name='gamma')

        self.beta = self.add_weight(shape=[self.n_class, c],
                                    initializer='zeros',
                                    trainable=True, name='beta')

        self.moving_mean = self.add_weight(shape=[1, 1, 1, c], initializer='zeros',
                                           trainable=False, name='moving_mean')

        self.moving_var = self.add_weight(shape=[1, 1, 1, c], initializer='ones',
                                          trainable=False, name='moving_var')

    def call(self, x, labels, training=False):

        beta = tf.gather(self.beta, labels)
        beta = tf.expand_dims(beta, 1)
        gamma = tf.gather(self.gamma, labels)
        gamma = tf.expand_dims(gamma, 1)

        if training:
            mean, var = tf.nn.moments(x, axes=(0, 1, 2), keepdims=True)
            self.moving_mean.assign(self.decay * self.moving_mean + (1 - self.decay) * mean)
            self.moving_var.assign(self.decay * self.moving_var + (1 - self.decay) * var)
            output = tf.nn.batch_normalization(x, mean, var, beta, gamma, self.eps)

        else:
            output = tf.nn.batch_normalization(x,
                                               self.moving_mean, self.moving_var,
                                               beta, gamma, self.eps)

        return output
