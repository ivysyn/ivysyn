# FingerprintOp

import tensorflow as tf

data = tf.constant(0, shape=[1,10], dtype=tf.int64)
method = tf.constant("farmhash64", shape=[], dtype=tf.string)
tf.raw_ops.Fingerprint(data=data, method=method)
