# ReverseOp

import tensorflow as tf

tensor = tf.constant(3, shape=[], dtype=tf.uint8)
dims = tf.constant(True, shape=[], dtype=tf.bool)
tf.raw_ops.Reverse(tensor=tensor, dims=dims)
