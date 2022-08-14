# QuantizedAvgPoolingOp

import tensorflow as tf

ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = "SAME"
input = tf.constant(1, shape=[1,4,4,2], dtype=tf.quint8)
min_input = tf.constant(0, shape=[1], dtype=tf.float32)
max_input = tf.constant(255, shape=[1], dtype=tf.float32)
tf.raw_ops.QuantizedAvgPool(input=input, min_input=min_input, max_input=max_input, ksize=ksize, strides=strides, padding=padding)
