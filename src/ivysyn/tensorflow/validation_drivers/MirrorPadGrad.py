# MirrorPadGradOp

import tensorflow as tf

mode = "SYMMETRIC"
input = tf.constant(1, shape=[3,7], dtype=tf.float32)
paddings = tf.constant(1, shape=[2,2], dtype=tf.int32)
tf.raw_ops.MirrorPadGrad(input=input, paddings=paddings, mode=mode)
