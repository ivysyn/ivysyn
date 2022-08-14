# QuantizedInstanceNorm

import tensorflow as tf

output_range_given = False
given_y_min = 0
given_y_max = 0
variance_epsilon = 1e-05
min_separation = 0.001
x = tf.constant(88, shape=[1,4,4,32], dtype=tf.quint8)
x_min = tf.constant(0, shape=[], dtype=tf.float32)
x_max = tf.constant(1, shape=[], dtype=tf.float32)
tf.raw_ops.QuantizedInstanceNorm(x=x, x_min=x_min, x_max=x_max, output_range_given=output_range_given, given_y_min=given_y_min, given_y_max=given_y_max, variance_epsilon=variance_epsilon, min_separation=min_separation)
