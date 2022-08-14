# IdentityNOp

import tensorflow as tf

input = tf.constant(2, shape=[1], dtype=tf.float32)
tf.raw_ops.IdentityN(input=input)
