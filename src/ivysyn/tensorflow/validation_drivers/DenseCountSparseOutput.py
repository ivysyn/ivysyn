# DenseCount

import tensorflow as tf

binary_output = False
minlength = 6
maxlength = 6
values = tf.constant(1, shape=[2,3], dtype=tf.int32)
weights = tf.constant([], shape=[0], dtype=tf.int32)
tf.raw_ops.DenseCountSparseOutput(values=values, weights=weights, binary_output=binary_output, minlength=minlength, maxlength=maxlength)
