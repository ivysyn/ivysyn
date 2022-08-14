# EnsureShapeOp

import tensorflow as tf

shape = [2,1]
input = tf.constant(.333333333333333310, shape=[2,1], dtype=tf.float64)
tf.raw_ops.EnsureShape(input=input, shape=shape)
