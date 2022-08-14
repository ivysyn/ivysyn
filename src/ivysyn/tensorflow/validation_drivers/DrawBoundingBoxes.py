# DrawBoundingBoxesOp

import tensorflow as tf

images = tf.constant(111, shape=[1, 4, 4, 1], dtype=tf.float32)
boxes = tf.constant(0, shape=[1, 1, 4], dtype=tf.float32)
tf.raw_ops.DrawBoundingBoxes(images=images, boxes=boxes)
