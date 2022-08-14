# DepthwiseConv2dNativeBackpropInputOp

import tensorflow as tf

strides = [1, 1, 1, 1]
padding = "SAME"
explicit_paddings = []
data_format = "NHWC"
dilations = [1, 1, 1, 1]
input_sizes = tf.constant(2, shape=[4], dtype=tf.int32)
filter = tf.constant(0.671319842, shape=[3,3,9,2], dtype=tf.float32)
out_backprop = tf.constant(0.558497429, shape=[2,5,5,18], dtype=tf.float32)
tf.raw_ops.DepthwiseConv2dNativeBackpropInput(input_sizes=input_sizes, filter=filter, out_backprop=out_backprop, strides=strides, padding=padding, explicit_paddings=explicit_paddings, data_format=data_format, dilations=dilations)
