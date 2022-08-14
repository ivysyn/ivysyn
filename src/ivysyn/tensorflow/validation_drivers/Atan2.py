# BinaryOp

import tensorflow as tf

y = tf.constant(1.41421354, shape=[], dtype=tf.float32)
x = tf.constant(-1.41421354, shape=[], dtype=tf.float32)
tf.raw_ops.Atan2(y=y, x=x)
