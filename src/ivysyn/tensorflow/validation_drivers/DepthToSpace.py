# DepthToSpaceOp

import tensorflow as tf

block_size = 2
data_format = "NHWC"
input = tf.constant(0.441032171, shape=[6,2,2,12], dtype=tf.float32)
tf.raw_ops.DepthToSpace(input=input, block_size=block_size, data_format=data_format)
