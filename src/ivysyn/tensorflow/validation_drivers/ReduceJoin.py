# ReduceJoinOp

import tensorflow as tf

keep_dims = False
separator = ", "
inputs = tf.constant("3", shape=[1], dtype=tf.string)
reduction_indices = tf.constant(0, shape=[1], dtype=tf.int32)
tf.raw_ops.ReduceJoin(inputs=inputs, reduction_indices=reduction_indices, keep_dims=keep_dims, separator=separator)
