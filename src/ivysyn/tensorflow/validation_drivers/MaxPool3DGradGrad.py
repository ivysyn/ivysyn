# MaxPooling3dGradGradOp

import tensorflow as tf

ksize = [1, 1, 3, 3, 1]
strides = [1, 1, 2, 2, 1]
padding = "VALID"
data_format = "NDHWC"
orig_input = tf.constant(0.228209138, shape=[9,2,12,12,3], dtype=tf.float32)
orig_output = tf.constant(0.92225194, shape=[9,2,5,5,3], dtype=tf.float32)
grad = tf.constant(1, shape=[9,2,12,12,3], dtype=tf.float32)
tf.raw_ops.MaxPool3DGradGrad(orig_input=orig_input, orig_output=orig_output, grad=grad, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
