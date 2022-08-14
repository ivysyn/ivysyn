# SqueezeOp

import tensorflow as tf

input = tf.constant(12, shape=[2,1], dtype=tf.int64)
tf.raw_ops.Squeeze(input=input)
