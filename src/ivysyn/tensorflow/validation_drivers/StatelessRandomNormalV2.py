# StatelessRandomOpBase

import tensorflow as tf

dtype = tf.int64
shape = tf.constant(1, shape=[1], dtype=tf.int32)
key = tf.constant(0, shape=[1], dtype=tf.uint64)
counter = tf.constant(0, shape=[2], dtype=tf.uint64)
alg = tf.constant(1, shape=[], dtype=tf.int32)
tf.raw_ops.StatelessRandomNormalV2(
    shape=shape, key=key, counter=counter, alg=alg)
