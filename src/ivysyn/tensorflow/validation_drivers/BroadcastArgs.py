# BCastArgsOp

import tensorflow as tf

s0 = tf.constant(5, shape=[1], dtype=tf.int64)
s1 = tf.constant(1, shape=[1], dtype=tf.int64)
tf.raw_ops.BroadcastArgs(s0=s0, s1=s1)
