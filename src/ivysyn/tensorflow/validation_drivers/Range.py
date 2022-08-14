# RangeOp

import tensorflow as tf

start = tf.constant(0, shape=[], dtype=tf.int64)
limit = tf.constant(10, shape=[], dtype=tf.int64)
delta = tf.constant(1, shape=[], dtype=tf.int64)
tf.raw_ops.Range(start=start, limit=limit, delta=delta)
