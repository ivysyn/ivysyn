# DilationBackpropFilterOp

import tensorflow as tf

strides = [1, 1, 1, 1]
rates = [1, 1, 1, 1]
padding = "VALID"
input = tf.constant(.4170220, shape=[1,3,3,1], dtype=tf.float32)
filter = tf.constant(0.53881675, shape=[1,1,1], dtype=tf.float32)
out_backprop = tf.constant(100, shape=[1,3,3,1], dtype=tf.float32)
tf.raw_ops.Dilation2DBackpropFilter(input=input, filter=filter, out_backprop=out_backprop, strides=strides, rates=rates, padding=padding)
