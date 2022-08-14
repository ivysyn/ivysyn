# ExtractImagePatchesOp

import tensorflow as tf

ksizes = [1, 5, 5, 1]
strides = [1, 1, 1, 1]
rates = [1, 1, 1, 1]
padding = "SAME"
images = tf.constant(.184634332179032020, shape=[4,512,512,1], dtype=tf.float64)
tf.raw_ops.ExtractImagePatches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)
