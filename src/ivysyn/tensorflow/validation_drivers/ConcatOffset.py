# ConcatOffsetOp

import tensorflow as tf

concat_dim = tf.constant(2, shape=[], dtype=tf.int32)
shape = tf.constant([32,10,10], shape=[3], dtype=tf.int32)
tf.raw_ops.ConcatOffset(concat_dim=concat_dim, shape=shape)
