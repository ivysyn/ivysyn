# MaxPoolingGradGradOp

import tensorflow as tf

ksize = [1, 3, 3, 1]
strides = [1, 2, 2, 1]
padding = "VALID"
data_format = "NHWC"
orig_input = tf.constant(0.786981, shape=[2,12,12,3], dtype=tf.float32)
orig_output = tf.constant(0.921271563, shape=[2,5,5,3], dtype=tf.float32)
grad = tf.constant(1, shape=[2,12,12,3], dtype=tf.float32)
tf.raw_ops.MaxPoolGradGradV2(orig_input=orig_input, orig_output=orig_output, grad=grad, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
