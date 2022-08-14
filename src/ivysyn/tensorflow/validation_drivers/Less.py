# BinaryOp

import tensorflow as tf

x = tf.constant(1.41421354, shape=[], dtype=tf.float32)
y = tf.constant(-1.41421354, shape=[], dtype=tf.float32)
tf.raw_ops.Less(x=x, y=y)
