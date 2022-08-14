# DiagPartOp

import tensorflow as tf

input = tf.constant(-2.74679446, shape=[2,2], dtype=tf.float32)
tf.raw_ops.DiagPart(input=input)
