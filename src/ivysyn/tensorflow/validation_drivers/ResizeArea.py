# ResizeAreaOp

import tensorflow as tf

align_corners = False
images = tf.constant(2.12024665, shape=[2,5,8,3], dtype=tf.float32)
size = tf.constant([2,2], shape=[2], dtype=tf.int32)
tf.raw_ops.ResizeArea(images=images, size=size, align_corners=align_corners)
