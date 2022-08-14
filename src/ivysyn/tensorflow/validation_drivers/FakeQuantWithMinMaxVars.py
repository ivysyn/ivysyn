# FakeQuantWithMinMaxVarsOp

import tensorflow as tf

num_bits = 8
narrow_range = False
inputs = tf.constant(0, shape=[2,3], dtype=tf.float32)
min = tf.constant(0, shape=[], dtype=tf.float32)
max = tf.constant(0, shape=[], dtype=tf.float32)
tf.raw_ops.FakeQuantWithMinMaxVars(inputs=inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range)
