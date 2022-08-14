# UnaryOp

import tensorflow as tf

x = tf.constant(False, shape=[5, 2], dtype=tf.bool)
tf.raw_ops.LogicalNot(x=x)
