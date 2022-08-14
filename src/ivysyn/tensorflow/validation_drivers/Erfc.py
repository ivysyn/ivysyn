# UnaryOp

import tensorflow as tf

x = tf.constant(0.23656249, shape=[5,2], dtype=tf.float32)
tf.raw_ops.Erfc(x=x)
