# BoostedTreesMakeQuantileSummariesOp

import tensorflow as tf

float_values = tf.constant(1.2, shape=[6], dtype=tf.float32)
example_weights = tf.constant(2.3, shape=[6], dtype=tf.float32)
epsilon = tf.constant(10, shape=[6], dtype=tf.float32)
tf.raw_ops.BoostedTreesMakeQuantileSummaries(float_values=float_values, example_weights=example_weights, epsilon=epsilon)
