# ScaleAndTranslateGradOp

import tensorflow as tf

kernel_type = "lanczos1"
antialias = True
grads = tf.constant(100, shape=[1,4,6,1], dtype=tf.float32)
original_image = tf.constant(12, shape=[1,2,3,1], dtype=tf.float32)
scale = tf.constant([1,1], shape=[2], dtype=tf.float32)
translation = tf.constant([0,0], shape=[2], dtype=tf.float32)
tf.raw_ops.ScaleAndTranslateGrad(grads=grads, original_image=original_image, scale=scale, translation=translation, kernel_type=kernel_type, antialias=antialias)
