# AssertOp

import tensorflow as tf

summarize = 3
condition = tf.constant(False, shape=[], dtype=tf.bool)
data = [tf.constant("fields", shape=[], dtype=tf.string)]
tf.raw_ops.Assert(condition=condition, data=data, summarize=summarize)
