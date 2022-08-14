# ResizeNearestNeighborOpGrad

import tensorflow as tf

align_corners = True
half_pixel_centers = False
grads = tf.constant(1, shape=[1,8,16,3], dtype=tf.float16)
size = tf.constant([4,6], shape=[2], dtype=tf.int32)
tf.raw_ops.ResizeNearestNeighborGrad(grads=grads, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
