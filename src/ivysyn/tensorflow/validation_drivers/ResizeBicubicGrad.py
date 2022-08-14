# ResizeBicubicOpGrad

import tensorflow as tf

align_corners = True
half_pixel_centers = False
grads = tf.constant(100, shape=[1,4,6,1], dtype=tf.float32)
original_image = tf.constant(12, shape=[1,2,3,1], dtype=tf.float32)
tf.raw_ops.ResizeBicubicGrad(grads=grads, original_image=original_image, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
