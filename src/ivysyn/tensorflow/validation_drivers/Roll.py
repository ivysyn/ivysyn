# RollOp

import tensorflow as tf

input = tf.constant(0.781604409, shape=[6,7], dtype=tf.float32)
shift = tf.constant(3, shape=[], dtype=tf.int32)
axis = tf.constant(1, shape=[], dtype=tf.int32)
tf.raw_ops.Roll(input=input, shift=shift, axis=axis)
