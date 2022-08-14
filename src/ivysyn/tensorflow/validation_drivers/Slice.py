# SliceOp

import tensorflow as tf

input = tf.constant(1, shape=[10], dtype=tf.int64)
begin = tf.constant(0, shape=[1], dtype=tf.int32)
size = tf.constant(10, shape=[1], dtype=tf.int32)
tf.raw_ops.Slice(input=input, begin=begin, size=size)
