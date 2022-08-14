# CopyOpBase

import tensorflow as tf

input = tf.constant(1, shape=[7,3], dtype=tf.float32)
tf.raw_ops.Copy(input=input)
