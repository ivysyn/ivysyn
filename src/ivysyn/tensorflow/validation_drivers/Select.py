# SelectOp

import tensorflow as tf

condition = tf.constant(False, shape=[], dtype=tf.bool)
x = tf.constant(".part", shape=[], dtype=tf.string)
y = tf.constant("_temp/part", shape=[], dtype=tf.string)
tf.raw_ops.Select(condition=condition, x=x, y=y)
