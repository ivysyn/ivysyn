# PadOp

import tensorflow as tf

input = tf.constant(1, shape=[3, 5], dtype=tf.float32)
paddings = tf.constant(0, shape=[2, 2], dtype=tf.int32)
tf.raw_ops.Pad(input=input, paddings=paddings)
