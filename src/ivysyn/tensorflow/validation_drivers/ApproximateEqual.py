# ApproximateEqualOp

import tensorflow as tf

tolerance = 0.3
x = tf.constant(3, shape=[], dtype=tf.float32)
y = tf.constant(2.9, shape=[], dtype=tf.float32)
tf.raw_ops.ApproximateEqual(x=x, y=y, tolerance=tolerance)
