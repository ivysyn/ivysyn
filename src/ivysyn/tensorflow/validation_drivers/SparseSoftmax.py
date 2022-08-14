# SparseSoftmaxOp

import tensorflow as tf

sp_indices = tf.constant(0, shape=[72,3], dtype=tf.int64)
sp_values = tf.constant(0.301107913, shape=[72], dtype=tf.float32)
sp_shape = tf.constant([1,6,12], shape=[3], dtype=tf.int64)
tf.raw_ops.SparseSoftmax(sp_indices=sp_indices, sp_values=sp_values, sp_shape=sp_shape)
