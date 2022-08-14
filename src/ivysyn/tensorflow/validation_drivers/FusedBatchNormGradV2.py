# FusedBatchNormGradOp

import tensorflow as tf

epsilon = 0.001
data_format = "NHWC"
is_training = True
y_backprop = tf.constant(2, shape=[1,1,6,2], dtype=tf.float32)
x = tf.constant(1, shape=[1,1,6,2], dtype=tf.float32)
scale = tf.constant([4,4], shape=[2], dtype=tf.float32)
reserve_space_1 = tf.constant([1.833,1.833], shape=[2], dtype=tf.float32)
reserve_space_2 = tf.constant([57.472,57.472], shape=[2], dtype=tf.float32)
tf.raw_ops.FusedBatchNormGradV2(y_backprop=y_backprop, x=x, scale=scale, reserve_space_1=reserve_space_1, reserve_space_2=reserve_space_2, epsilon=epsilon, data_format=data_format, is_training=is_training)
