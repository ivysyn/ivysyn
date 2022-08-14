# ResizeBilinearOp

import tensorflow as tf

align_corners = False
half_pixel_centers = True
images = tf.constant(0.184634328, shape=[2,5,8,3], dtype=tf.float32)
size = tf.constant([4,8], shape=[2], dtype=tf.int32)
tf.raw_ops.ResizeBilinear(images=images, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
