# DepthwiseConv2dNativeBackpropFilterOp

import tensorflow as tf

strides = [1, 2, 2, 1]
padding = "VALID"
explicit_paddings = []
data_format = "NHWC"
dilations = [1, 1, 1, 1]
input = tf.constant(0.463723898, shape=[2,12,12,3], dtype=tf.float32)
filter_sizes = tf.constant(3, shape=[4], dtype=tf.int32)
out_backprop = tf.constant(0.828671336, shape=[2,5,5,6], dtype=tf.float32)
tf.raw_ops.DepthwiseConv2dNativeBackpropFilter(input=input, filter_sizes=filter_sizes, out_backprop=out_backprop, strides=strides, padding=padding, explicit_paddings=explicit_paddings, data_format=data_format, dilations=dilations)
