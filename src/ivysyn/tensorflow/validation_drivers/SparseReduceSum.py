# SparseReduceOp

import tensorflow as tf

keep_dims = False
input_indices = tf.constant(0, shape=[114,5], dtype=tf.int64)
input_values = tf.constant(1.6566996986258962, shape=[114], dtype=tf.float64)
input_shape = tf.constant(11, shape=[5], dtype=tf.int64)
reduction_axes = tf.constant(0, shape=[1], dtype=tf.int32)
tf.raw_ops.SparseReduceSum(input_indices=input_indices, input_values=input_values, input_shape=input_shape, reduction_axes=reduction_axes, keep_dims=keep_dims)
