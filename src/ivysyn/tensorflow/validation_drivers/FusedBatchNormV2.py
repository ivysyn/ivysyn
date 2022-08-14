# FusedBatchNormOp

import tensorflow as tf

epsilon = 0.0001
exponential_avg_factor = 1
data_format = "NHWC"
is_training = True
x = tf.constant([], shape=[0,1,1,1], dtype=tf.float32)
scale = tf.constant(1, shape=[1], dtype=tf.float32)
offset = tf.constant(1, shape=[1], dtype=tf.float32)
mean = tf.constant([], shape=[0], dtype=tf.float32)
variance = tf.constant([], shape=[0], dtype=tf.float32)
tf.raw_ops.FusedBatchNormV2(x=x, scale=scale, offset=offset, mean=mean, variance=variance, epsilon=epsilon, exponential_avg_factor=exponential_avg_factor, data_format=data_format, is_training=is_training)
