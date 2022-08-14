# BCastGradArgsOp

import tensorflow as tf

s0 = tf.constant([], shape=[0], dtype=tf.int32)
s1 = tf.constant([], shape=[0], dtype=tf.int32)
tf.raw_ops.BroadcastGradientArgs(s0=s0, s1=s1)
