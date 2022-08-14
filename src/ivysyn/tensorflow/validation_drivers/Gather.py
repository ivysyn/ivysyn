# GatherOp

import tensorflow as tf

params = tf.constant(111, shape=[10,1], dtype=tf.float32)
indices = tf.constant(1, shape=[5], dtype=tf.int64)
validate_indices = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.Gather(params=params, indices=indices, validate_indices=validate_indices)
