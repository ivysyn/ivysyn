# LinSpaceOp

import tensorflow as tf

start = tf.constant(3, shape=[], dtype=tf.float32)
stop = tf.constant(7, shape=[], dtype=tf.float32)
num = tf.constant(3, shape=[], dtype=tf.int32)
tf.raw_ops.LinSpace(start=start, stop=stop, num=num)
