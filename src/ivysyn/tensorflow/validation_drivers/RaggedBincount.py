# RaggedBincountOp

import tensorflow as tf

binary_output = True
splits = tf.constant([0,3,5], shape=[3], dtype=tf.int64)
values = tf.constant(1, shape=[5], dtype=tf.int32)
size = tf.constant(6, shape=[], dtype=tf.int32)
weights = tf.constant([], shape=[0], dtype=tf.float32)
tf.raw_ops.RaggedBincount(splits=splits, values=values, size=size, weights=weights, binary_output=binary_output)
