# BinaryOp

import tensorflow as tf

a = tf.constant(1.41421354, shape=[], dtype=tf.float32)
x = tf.constant(-1.41421354, shape=[], dtype=tf.float32)
tf.raw_ops.Igammac(a=a, x=x)
