# SpaceToDepthOp

import tensorflow as tf

block_size = 2
data_format = "NHWC"
input = tf.constant(0, shape=[7,10,16,5], dtype=tf.float32)
tf.raw_ops.SpaceToDepth(input=input, block_size=block_size, data_format=data_format)
