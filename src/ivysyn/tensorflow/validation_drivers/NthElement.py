# NthElementOp

import tensorflow as tf

reverse = False
input = tf.constant(0.184634328, shape=[1], dtype=tf.float32)
n = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.NthElement(input=input, n=n, reverse=reverse)
