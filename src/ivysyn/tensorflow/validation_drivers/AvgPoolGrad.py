# AvgPoolingGradOp

import tensorflow as tf

ksize = [1, 1, 1, 1]
strides = [1, 1, 1, 1]
padding = "VALID"
data_format = "NHWC"
orig_input_shape = tf.constant(6, shape=[4], dtype=tf.int32)
grad = tf.constant(0.430592865, shape=[6,5,5,3], dtype=tf.float32)
tf.raw_ops.AvgPoolGrad(orig_input_shape=orig_input_shape, grad=grad, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
