# QuantizedConv2DOp

import tensorflow as tf

strides = [1, 1, 1, 1]
padding = "VALID"
out_type = tf.qint32
dilations = [1, 1, 1, 1]
input = tf.constant(1, shape=[1,2,3,3], dtype=tf.quint8)
filter = tf.constant(1, shape=[1,1,3,3], dtype=tf.quint8)
min_input = tf.constant(0, shape=[], dtype=tf.float32)
max_input = tf.constant(255, shape=[], dtype=tf.float32)
min_filter = tf.constant(0, shape=[], dtype=tf.float32)
max_filter = tf.constant(255, shape=[], dtype=tf.float32)
tf.raw_ops.QuantizedConv2D(input=input, filter=filter, min_input=min_input, max_input=max_input, min_filter=min_filter, max_filter=max_filter, strides=strides, padding=padding, out_type=out_type, dilations=dilations)
