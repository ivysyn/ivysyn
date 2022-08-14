# BinaryOp

import tensorflow as tf

x = tf.constant(1.41421354, shape=[], dtype=tf.float32)
q = tf.constant(-1.41421354, shape=[], dtype=tf.float32)
tf.raw_ops.Zeta(x=x, q=q)
