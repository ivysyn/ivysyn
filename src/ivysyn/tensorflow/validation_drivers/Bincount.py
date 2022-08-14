# BincountOp

import tensorflow as tf

arr = tf.constant([0,0,1], shape=[3], dtype=tf.int32)
size = tf.constant(2, shape=[], dtype=tf.int32)
weights = tf.constant([], shape=[0], dtype=tf.int64)
tf.raw_ops.Bincount(arr=arr, size=size, weights=weights)
