# BinaryOp

import tensorflow as tf

x = tf.constant([-2, 64, 101, 32], dtype=tf.int8)
y = tf.constant([-1, -5, -3, -14], dtype=tf.int8)

tf.raw_ops.RightShift(x=x, y=y)
