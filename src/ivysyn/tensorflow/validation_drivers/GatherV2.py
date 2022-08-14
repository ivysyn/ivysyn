# GatherOp

import tensorflow as tf

batch_dims = 0
params = tf.constant(111, shape=[10,1], dtype=tf.float32)
indices = tf.constant(1, shape=[5], dtype=tf.int64)
axis = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.GatherV2(params=params, indices=indices, axis=axis, batch_dims=batch_dims)
