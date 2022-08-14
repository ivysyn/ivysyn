# SparseSparseBinaryOpShared

import tensorflow as tf

a_indices = tf.constant(0, shape=[1,1], dtype=tf.int64)
a_values = tf.constant(0, shape=[1], dtype=tf.int32)
a_shape = tf.constant(7, shape=[1], dtype=tf.int64)
b_indices = tf.constant(0, shape=[1,1], dtype=tf.int64)
b_values = tf.constant(1, shape=[1], dtype=tf.int32)
b_shape = tf.constant(7, shape=[1], dtype=tf.int64)
tf.raw_ops.SparseSparseMinimum(a_indices=a_indices, a_values=a_values, a_shape=a_shape, b_indices=b_indices, b_values=b_values, b_shape=b_shape)
