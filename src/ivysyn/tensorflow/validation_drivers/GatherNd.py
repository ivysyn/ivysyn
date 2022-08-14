# GatherNdOp

import tensorflow as tf

params = tf.constant(1, shape=[6,12], dtype=tf.float32)
indices = tf.constant(0, shape=[17,2], dtype=tf.int64)
tf.raw_ops.GatherNd(params=params, indices=indices)
