# CopyOpBase

import tensorflow as tf

x = tf.constant(1, shape=[7,3], dtype=tf.float32)
tf.raw_ops.DeepCopy(x=x)
