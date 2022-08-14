# EncodeJpegVariableQualityOp

import tensorflow as tf

images = tf.constant(0, shape=[2,4,3], dtype=tf.uint8)
quality = tf.constant(80, shape=[], dtype=tf.int32)
tf.raw_ops.EncodeJpegVariableQuality(images=images, quality=quality)
