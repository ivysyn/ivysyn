# WhereCPUOp

import tensorflow as tf

condition = tf.constant(True, shape=[3], dtype=tf.bool)
tf.raw_ops.Where(condition=condition)
