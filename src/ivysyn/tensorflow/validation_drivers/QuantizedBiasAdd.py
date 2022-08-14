# QuantizedBiasAddOp

import tensorflow as tf

out_type = tf.qint32
input = tf.constant(43, shape=[2,3], dtype=tf.quint8)
bias = tf.constant([85,170,255], shape=[3], dtype=tf.quint8)
min_input = tf.constant(0, shape=[1], dtype=tf.float32)
max_input = tf.constant(60, shape=[1], dtype=tf.float32)
min_bias = tf.constant(0, shape=[1], dtype=tf.float32)
max_bias = tf.constant(3, shape=[1], dtype=tf.float32)
tf.raw_ops.QuantizedBiasAdd(input=input, bias=bias, min_input=min_input, max_input=max_input, min_bias=min_bias, max_bias=max_bias, out_type=out_type)
