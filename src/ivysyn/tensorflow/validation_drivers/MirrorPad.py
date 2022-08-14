# MirrorPadOp

import tensorflow as tf

mode = "REFLECT"
input = tf.constant(1, shape=[1,2,3,2], dtype=tf.float32)
paddings = tf.constant(0, shape=[4,2], dtype=tf.int32)
tf.raw_ops.MirrorPad(input=input, paddings=paddings, mode=mode)
