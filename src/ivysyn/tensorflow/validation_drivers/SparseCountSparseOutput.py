# SparseCount

import tensorflow as tf

binary_output = False
minlength = -1
maxlength = -1
indices = tf.constant(0, shape=[5,2], dtype=tf.int64)
values = tf.constant(0, shape=[5], dtype=tf.int64)
dense_shape = tf.constant([4,2], shape=[2], dtype=tf.int64)
weights = tf.constant([], shape=[0], dtype=tf.int64)
tf.raw_ops.SparseCountSparseOutput(indices=indices, values=values, dense_shape=dense_shape, weights=weights, binary_output=binary_output, minlength=minlength, maxlength=maxlength)
