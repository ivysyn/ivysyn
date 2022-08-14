# RGBToHSVOp

import tensorflow as tf

images = tf.constant(0.548813522, shape=[1,5,5,3], dtype=tf.float32)
tf.raw_ops.RGBToHSV(images=images)
