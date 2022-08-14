# SplitOpCPU

import tensorflow as tf

num_split = 4
axis = tf.constant(1, shape=[], dtype=tf.int32)
value = tf.constant(0.306165963, shape=[3,40], dtype=tf.float32)
tf.raw_ops.Split(axis=axis, value=value, num_split=num_split)
