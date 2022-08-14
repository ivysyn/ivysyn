# FusedResizeConv2DUsingGemmOp

import tensorflow as tf

mode = "REFLECT"
strides = [1, 1, 1, 1]
padding = "SAME"
input = tf.constant(1, shape=[1,2,3,2], dtype=tf.float32)
paddings = tf.constant(0, shape=[4,2], dtype=tf.int32)
filter = tf.constant(1, shape=[1,2,2,2], dtype=tf.float32)
tf.raw_ops.FusedPadConv2D(input=input, paddings=paddings, filter=filter, mode=mode, strides=strides, padding=padding)
