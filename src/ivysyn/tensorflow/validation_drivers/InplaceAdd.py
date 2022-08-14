# InplaceOpBase

import tensorflow as tf

x = tf.constant(0, shape=[13,8], dtype=tf.float32)
i = tf.constant(12, shape=[1], dtype=tf.int32)
v = tf.constant(0, shape=[1,8], dtype=tf.float32)
tf.raw_ops.InplaceAdd(x=x, i=i, v=v)
