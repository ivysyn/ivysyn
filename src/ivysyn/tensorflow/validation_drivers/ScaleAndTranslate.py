# ScaleAndTranslateOp

import tensorflow as tf

kernel_type = "lanczos1"
antialias = True
images = tf.constant(12, shape=[1,2,3,1], dtype=tf.float32)
size = tf.constant([4,6], shape=[2], dtype=tf.int32)
scale = tf.constant([1,1], shape=[2], dtype=tf.float32)
translation = tf.constant([0,0], shape=[2], dtype=tf.float32)
tf.raw_ops.ScaleAndTranslate(images=images, size=size, scale=scale, translation=translation, kernel_type=kernel_type, antialias=antialias)
