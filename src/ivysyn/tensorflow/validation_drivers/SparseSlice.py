# SparseSliceOp

import tensorflow as tf

indices = tf.constant(0, shape=[4,2], dtype=tf.int64)
values = tf.constant(2, shape=[4], dtype=tf.int32)
shape = tf.constant([2,5], shape=[2], dtype=tf.int64)
start = tf.constant([0,0], shape=[2], dtype=tf.int64)
size = tf.constant([2,3], shape=[2], dtype=tf.int64)
tf.raw_ops.SparseSlice(indices=indices, values=values, shape=shape, start=start, size=size)
