# SparseBincountOp

import tensorflow as tf

binary_output = True
indices = tf.constant(0, shape=[5,2], dtype=tf.int64)
values = tf.constant(0, shape=[5], dtype=tf.int64)
dense_shape = tf.constant([4,2], shape=[2], dtype=tf.int64)
size = tf.constant(6, shape=[], dtype=tf.int64)
weights = tf.constant([], shape=[0], dtype=tf.float32)
tf.raw_ops.SparseBincount(indices=indices, values=values, dense_shape=dense_shape, size=size, weights=weights, binary_output=binary_output)
