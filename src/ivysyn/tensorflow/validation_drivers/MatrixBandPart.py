# MatrixBandPartOp

import tensorflow as tf

input = tf.constant(-1, shape=[2, 2], dtype=tf.float32)
num_lower = tf.constant(-1, shape=[], dtype=tf.int64)
num_upper = tf.constant(0, shape=[], dtype=tf.int64)
tf.raw_ops.MatrixBandPart(
    input=input, num_lower=num_lower, num_upper=num_upper)
