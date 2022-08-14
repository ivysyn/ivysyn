# SparseDenseBinaryOpShared

import tensorflow as tf

sp_indices = tf.constant(0, shape=[3,2], dtype=tf.int64)
sp_values = tf.constant([42,43,44], shape=[3], dtype=tf.int64)
sp_shape = tf.constant([2,2], shape=[2], dtype=tf.int64)
dense = tf.constant(2, shape=[], dtype=tf.int64)
tf.raw_ops.SparseDenseCwiseAdd(sp_indices=sp_indices, sp_values=sp_values, sp_shape=sp_shape, dense=dense)
