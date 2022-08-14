# ReductionOp

import tensorflow as tf

keep_dims = False
input = tf.constant(111, shape=[5,1], dtype=tf.float32)
axis = tf.constant([0,1], shape=[2], dtype=tf.int32)
tf.raw_ops.EuclideanNorm(input=input, axis=axis, keep_dims=keep_dims)
