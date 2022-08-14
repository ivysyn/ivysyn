# UnaryOp

import tensorflow as tf

x = tf.constant(1, shape=[5, 2], dtype=tf.int32)
tf.raw_ops.Invert(x=x)
