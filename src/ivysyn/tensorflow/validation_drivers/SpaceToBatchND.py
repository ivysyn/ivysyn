# SpaceToBatchNDOp

import tensorflow as tf

input = tf.constant(123, shape=[2,2,4,1], dtype=tf.float32)
block_shape = tf.constant([2,2], shape=[2], dtype=tf.int64)
paddings = tf.constant(0, shape=[2,2], dtype=tf.int32)
tf.raw_ops.SpaceToBatchND(input=input, block_shape=block_shape, paddings=paddings)
