# AdjustSaturationOpBase

import tensorflow as tf

images = tf.constant(0.972403526, shape=[3,2,4,4,3], dtype=tf.float32)
scale = tf.constant(0.1, shape=[], dtype=tf.float32)
tf.raw_ops.AdjustSaturation(images=images, scale=scale)
