# StatelessRandomBinomialOp

import tensorflow as tf

dtype = tf.half
shape = tf.constant(1000, shape=[1], dtype=tf.int32)
seed = tf.constant([12,34], shape=[2], dtype=tf.int32)
counts = tf.constant(10, shape=[], dtype=tf.float32)
probs = tf.constant(0.4, shape=[], dtype=tf.float32)
tf.raw_ops.StatelessRandomBinomial(shape=shape, seed=seed, counts=counts, probs=probs, dtype=dtype)
