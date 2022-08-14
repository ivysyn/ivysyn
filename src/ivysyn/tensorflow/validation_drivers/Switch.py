# SwitchOp

import tensorflow as tf

data = tf.constant(2, shape=[], dtype=tf.float32)
pred = tf.constant(False, shape=[], dtype=tf.bool)
tf.raw_ops.Switch(data=data, pred=pred)
