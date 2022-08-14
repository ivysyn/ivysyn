# SparseFillEmptyRowsOp

import tensorflow as tf

indices = tf.constant(0, shape=[2,2], dtype=tf.int64)
values = tf.constant([0,1], shape=[2], dtype=tf.int64)
dense_shape = tf.constant([2,1], shape=[2], dtype=tf.int64)
default_value = tf.constant(0, shape=[], dtype=tf.int64)
tf.raw_ops.SparseFillEmptyRows(indices=indices, values=values, dense_shape=dense_shape, default_value=default_value)
