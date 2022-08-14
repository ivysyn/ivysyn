# SparseToDense

import tensorflow as tf

validate_indices = True
sparse_indices = tf.constant(0, shape=[2,2], dtype=tf.int64)
output_shape = tf.constant([2,1], shape=[2], dtype=tf.int64)
sparse_values = tf.constant([0,1], shape=[2], dtype=tf.int64)
default_value = tf.constant(-1, shape=[], dtype=tf.int64)
tf.raw_ops.SparseToDense(sparse_indices=sparse_indices, output_shape=output_shape, sparse_values=sparse_values, default_value=default_value, validate_indices=validate_indices)
