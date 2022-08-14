# BetaincOp

import tensorflow as tf

a = tf.constant(1.5, shape=[6], dtype=tf.float32)
b = tf.constant(0.5, shape=[], dtype=tf.float32)
x = tf.constant(0.680248, shape=[6], dtype=tf.float32)
tf.raw_ops.Betainc(a=a, b=b, x=x)
