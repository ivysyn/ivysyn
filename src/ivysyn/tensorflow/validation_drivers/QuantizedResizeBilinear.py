# QuantizedResizeBilinearOp

import tensorflow as tf

align_corners = False
half_pixel_centers = False
images = tf.constant(1, shape=[1,1,128,1], dtype=tf.qint32)
size = tf.constant([1,256], shape=[2], dtype=tf.int32)
min = tf.constant(0, shape=[], dtype=tf.float32)
max = tf.constant(256, shape=[], dtype=tf.float32)
tf.raw_ops.QuantizedResizeBilinear(images=images, size=size, min=min, max=max, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
