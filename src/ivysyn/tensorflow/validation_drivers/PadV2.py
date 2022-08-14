# PadOp

import tensorflow as tf

input = tf.constant(1, shape=[3, 5], dtype=tf.float32)
paddings = tf.constant(0, shape=[2, 2], dtype=tf.int32)
constant_values = tf.constant(0, shape=[], dtype=tf.float32)
tf.raw_ops.PadV2(input=input, paddings=paddings,
                 constant_values=constant_values)
