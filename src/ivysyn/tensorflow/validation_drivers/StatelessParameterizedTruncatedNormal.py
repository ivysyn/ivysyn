# StatelessParameterizedTruncatedNormal

import tensorflow as tf

shape = tf.constant(100000, shape=[1], dtype=tf.int32)
seed = tf.constant([732735241,2028384937], shape=[2], dtype=tf.int32)
means = tf.constant(0, shape=[], dtype=tf.float32)
stddevs = tf.constant(1, shape=[], dtype=tf.float32)
minvals = tf.constant(-2, shape=[], dtype=tf.float32)
maxvals = tf.constant(2, shape=[], dtype=tf.float32)
tf.raw_ops.StatelessParameterizedTruncatedNormal(shape=shape, seed=seed, means=means, stddevs=stddevs, minvals=minvals, maxvals=maxvals)
