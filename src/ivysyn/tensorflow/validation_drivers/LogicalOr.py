# BinaryOp

import tensorflow as tf

x = tf.constant(True, shape=[], dtype=tf.bool)
y = tf.constant(True, shape=[], dtype=tf.bool)
tf.raw_ops.LogicalOr(x=x, y=y)
