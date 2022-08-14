# ReshapeOp

import tensorflow as tf

tensor = tf.constant(1, shape=[10], dtype=tf.int64)
shape = tf.constant([2,5], shape=[2], dtype=tf.int32)
tf.raw_ops.Reshape(tensor=tensor, shape=shape)
