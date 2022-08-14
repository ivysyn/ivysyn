# ExpandDimsOp

import tensorflow as tf

input = tf.constant(8.39427503e-06, shape=[], dtype=tf.float32)
axis = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.ExpandDims(input=input, axis=axis)
