# ExtractVolumePatchesOp

import tensorflow as tf

ksizes = [1, 1, 1, 1, 1]
strides = [1, 1, 1, 1, 1]
padding = "VALID"
input = tf.constant(1, shape=[2,3,4,5,6], dtype=tf.float16)
tf.raw_ops.ExtractVolumePatches(input=input, ksizes=ksizes, strides=strides, padding=padding)
