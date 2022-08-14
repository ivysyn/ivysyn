# ImageProjectiveTransformV2

import tensorflow as tf

interpolation = "BILINEAR"
fill_mode = "REFLECT"
images = tf.constant(0.184634328, shape=[2,5,8,3], dtype=tf.float32)
transforms = tf.constant(0.378575385, shape=[2,8], dtype=tf.float32)
output_shape = tf.constant([5,8], shape=[2], dtype=tf.int32)
tf.raw_ops.ImageProjectiveTransformV2(images=images, transforms=transforms, output_shape=output_shape, interpolation=interpolation, fill_mode=fill_mode)
