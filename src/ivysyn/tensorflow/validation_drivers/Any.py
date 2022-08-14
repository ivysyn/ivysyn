# ReductionOp

import tensorflow as tf

keep_dims = False
input = tf.constant(True, shape=[5, 1], dtype=tf.bool)
axis = tf.constant([0, 1], shape=[2], dtype=tf.int32)
tf.raw_ops.Any(input=input, axis=axis, keep_dims=keep_dims)
