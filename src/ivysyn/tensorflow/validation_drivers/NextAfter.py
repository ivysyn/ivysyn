# BinaryOp

import tensorflow as tf

x1 = tf.constant(1.41421354, shape=[], dtype=tf.float32)
x2 = tf.constant(-1.41421354, shape=[], dtype=tf.float32)
tf.raw_ops.NextAfter(x1=x1, x2=x2)
