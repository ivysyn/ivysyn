# DilationOp

import tensorflow as tf

strides = [1, 1, 1, 1]
rates = [1, 1, 1, 1]
padding = "VALID"
input = tf.constant(.10, shape=[1,2,2,1], dtype=tf.float32)
filter = tf.constant(.40, shape=[2,2,1], dtype=tf.float32)
tf.raw_ops.Dilation2D(input=input, filter=filter, strides=strides, rates=rates, padding=padding)
