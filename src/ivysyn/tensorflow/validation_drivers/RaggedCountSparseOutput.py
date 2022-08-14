# RaggedCount

import tensorflow as tf

binary_output = False
minlength = 6
maxlength = 6
splits = tf.constant([0,3,5], shape=[3], dtype=tf.int64)
values = tf.constant(1, shape=[5], dtype=tf.int32)
weights = tf.constant([], shape=[0], dtype=tf.int32)
tf.raw_ops.RaggedCountSparseOutput(splits=splits, values=values, weights=weights, binary_output=binary_output, minlength=minlength, maxlength=maxlength)
