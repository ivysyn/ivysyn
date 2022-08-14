# SparseReorderOp

import tensorflow as tf

input_indices = tf.constant(5, shape=[5,2], dtype=tf.int64)
input_values = tf.constant(0.233435854, shape=[5], dtype=tf.float32)
input_shape = tf.constant([10,10], shape=[2], dtype=tf.int64)
tf.raw_ops.SparseReorder(input_indices=input_indices, input_values=input_values, input_shape=input_shape)
