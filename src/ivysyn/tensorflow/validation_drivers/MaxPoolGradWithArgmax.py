# MaxPoolingGradWithArgmaxOp

import tensorflow as tf

ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
padding = "VALID"
include_batch_in_index = False
input = tf.constant(111, shape=[2,3,3,1], dtype=tf.float32)
grad = tf.constant(111213, shape=[2,2,2,1], dtype=tf.float32)
argmax = tf.constant(13, shape=[2,2,2,1], dtype=tf.int64)
tf.raw_ops.MaxPoolGradWithArgmax(input=input, grad=grad, argmax=argmax, ksize=ksize, strides=strides, padding=padding, include_batch_in_index=include_batch_in_index)
