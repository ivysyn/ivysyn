# DequantizeOp

import tensorflow as tf

mode = "SCALED"
narrow_range = False
axis = -1
dtype = tf.float32
input = tf.constant(-128, shape=[2,3,4,5], dtype=tf.qint8)
min_range = tf.constant(-2, shape=[], dtype=tf.float32)
max_range = tf.constant(1.6, shape=[], dtype=tf.float32)
tf.raw_ops.Dequantize(input=input, min_range=min_range, max_range=max_range, mode=mode, narrow_range=narrow_range, axis=axis, dtype=dtype)
