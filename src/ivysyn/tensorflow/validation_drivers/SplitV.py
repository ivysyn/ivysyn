# SplitVOpCPU

import tensorflow as tf

num_split = 16
value = tf.constant(-4.25593134e+1, shape=[32, 10, 20], dtype=tf.float32)
size_splits = tf.constant([10, 10], shape=[2], dtype=tf.int32)
axis = tf.constant(2, shape=[], dtype=tf.int32)
tf.raw_ops.SplitV(value=value, size_splits=size_splits,
                  axis=axis, num_split=num_split)
