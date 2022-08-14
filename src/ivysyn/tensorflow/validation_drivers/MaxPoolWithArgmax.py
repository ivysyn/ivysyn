# MaxPoolingWithArgmaxOp

import tensorflow as tf

ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
padding = "VALID"
Targmax = tf.int64
include_batch_in_index = False
input = tf.constant(111, shape=[2,3,3,1], dtype=tf.float32)
tf.raw_ops.MaxPoolWithArgmax(input=input, ksize=ksize, strides=strides, padding=padding, Targmax=Targmax, include_batch_in_index=include_batch_in_index)
