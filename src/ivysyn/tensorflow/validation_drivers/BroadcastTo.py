# BroadcastToOp

import tensorflow as tf

input = tf.constant(.20, shape=[5,1], dtype=tf.float32)
shape = tf.constant([5,2], shape=[2], dtype=tf.int32)
tf.raw_ops.BroadcastTo(input=input, shape=shape)
