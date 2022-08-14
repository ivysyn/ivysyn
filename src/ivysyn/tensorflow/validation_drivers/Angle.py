# UnaryOp

import tensorflow as tf

input = tf.constant(0.23656249, shape=[5, 2], dtype=tf.complex64)
tf.raw_ops.Angle(input=input)
