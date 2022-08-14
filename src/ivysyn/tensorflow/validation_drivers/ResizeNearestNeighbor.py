# ResizeNearestNeighborOp

import tensorflow as tf

align_corners = False
half_pixel_centers = True
images = tf.constant(12, shape=[1,4,2,1], dtype=tf.int64)
size = tf.constant([2,2], shape=[2], dtype=tf.int32)
tf.raw_ops.ResizeNearestNeighbor(images=images, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
