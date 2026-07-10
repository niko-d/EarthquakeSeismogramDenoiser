import tensorflow as tf

class MaxAbsNorm1D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        # x: (B, T, C)
        m = tf.reduce_max(tf.abs(x), axis=1, keepdims=True)
        m = tf.maximum(m, self.eps)
        return x / m
