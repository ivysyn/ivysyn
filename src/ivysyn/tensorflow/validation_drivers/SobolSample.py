# SobolSampleOp

import tensorflow as tf

dtype = tf.float64
dim = tf.constant(2, shape=[], dtype=tf.int32)
num_results = tf.constant(4, shape=[], dtype=tf.int32)
skip = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.SobolSample(dim=dim, num_results=num_results, skip=skip, dtype=dtype)
