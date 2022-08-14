# RequantizationRangeOp

import tensorflow as tf

input = tf.constant([1], shape=[3], dtype=tf.qint32)
input_min = tf.constant(-256, shape=[1], dtype=tf.float32)
input_max = tf.constant(256, shape=[1], dtype=tf.float32)
tf.raw_ops.RequantizationRange(input=input, input_min=input_min, input_max=input_max)
