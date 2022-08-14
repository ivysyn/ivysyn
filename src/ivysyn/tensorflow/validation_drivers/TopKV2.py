# TopK

import tensorflow as tf

sorted = True
input = tf.constant(-0, shape=[1], dtype=tf.float64)
k = tf.constant(1, shape=[], dtype=tf.int32)
tf.raw_ops.TopKV2(input=input, k=k, sorted=sorted)
