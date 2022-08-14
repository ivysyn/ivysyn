# BatchToSpaceNDOp

import tensorflow as tf

input = tf.constant(139, shape=[8,1,2,1], dtype=tf.float32)
block_shape = tf.constant([2,2], shape=[2], dtype=tf.int64)
crops = tf.constant(0, shape=[2,2], dtype=tf.int32)
tf.raw_ops.BatchToSpaceND(input=input, block_shape=block_shape, crops=crops)
