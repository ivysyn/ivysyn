# Pooling3DOp

import tensorflow as tf

ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
padding = "VALID"
data_format = "NDHWC"
input = tf.constant(0.836992383, shape=[9,7,6,6,5], dtype=tf.float32)
tf.raw_ops.AvgPool3D(input=input, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
