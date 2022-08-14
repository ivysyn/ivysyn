# SparseReduceSparseOp

import tensorflow as tf

keep_dims = False
input_indices = tf.constant(0, shape=[4,2], dtype=tf.int64)
input_values = tf.constant(2, shape=[4], dtype=tf.float32)
input_shape = tf.constant([3,2], shape=[2], dtype=tf.int64)
reduction_axes = tf.constant(0, shape=[1], dtype=tf.int32)
tf.raw_ops.SparseReduceSumSparse(input_indices=input_indices, input_values=input_values, input_shape=input_shape, reduction_axes=reduction_axes, keep_dims=keep_dims)
