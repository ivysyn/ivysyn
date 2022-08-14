# BinaryOp

import tensorflow as tf

x = tf.constant(1, shape=[], dtype=tf.int32)
y = tf.constant(1, shape=[], dtype=tf.int32)
tf.raw_ops.TruncateDiv(x=x, y=y)
