# FusedBatchNormOpV3

import tensorflow as tf

epsilon = 0.001
exponential_avg_factor = 1
data_format = "NHWC"
is_training = False
x = tf.constant(4.87283707, shape=[3,2,4,2], dtype=tf.float32)
scale = tf.constant([1,1], shape=[2], dtype=tf.float32)
offset = tf.constant([1,1], shape=[2], dtype=tf.float32)
mean = tf.constant([0,0], shape=[2], dtype=tf.float32)
variance = tf.constant([1,1], shape=[2], dtype=tf.float32)
tf.raw_ops.FusedBatchNormV3(x=x, scale=scale, offset=offset, mean=mean, variance=variance, epsilon=epsilon, exponential_avg_factor=exponential_avg_factor, data_format=data_format, is_training=is_training)
