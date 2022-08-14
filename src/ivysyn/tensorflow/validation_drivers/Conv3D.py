# Conv3DOp

import tensorflow as tf

strides = [1, 1, 1, 1, 1]
padding = "VALID"
data_format = "NDHWC"
dilations = [1, 1, 1, 1, 1]
input = tf.constant(1, shape=[2,6,8,6,2], dtype=tf.float32)
filter = tf.constant(0.0445518792, shape=[3,3,3,2,2], dtype=tf.float32)
tf.raw_ops.Conv3D(input=input, filter=filter, strides=strides, padding=padding, data_format=data_format, dilations=dilations)
