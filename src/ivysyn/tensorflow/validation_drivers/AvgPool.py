# AvgPoolingOp

import tensorflow as tf

ksize = [1, 3, 3, 1]
strides = [1, 2, 2, 1]
padding = "VALID"
data_format = "NHWC"
value = tf.constant(0.051429987, shape=[6,12,12,3], dtype=tf.float32)
tf.raw_ops.AvgPool(value=value, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
