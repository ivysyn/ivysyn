# FillOp

import tensorflow as tf

dims = tf.constant([], shape=[0], dtype=tf.int32)
value = tf.constant(0, shape=[], dtype=tf.float32)
tf.raw_ops.Fill(dims=dims, value=value)
