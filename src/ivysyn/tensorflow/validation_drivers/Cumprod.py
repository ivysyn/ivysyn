# ScanOp

import tensorflow as tf

exclusive = False
reverse = False
x = tf.constant([3,4], shape=[2], dtype=tf.int64)
axis = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.Cumprod(x=x, axis=axis, exclusive=exclusive, reverse=reverse)
