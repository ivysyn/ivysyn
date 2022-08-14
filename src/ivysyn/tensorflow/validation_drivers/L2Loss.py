# L2LossOp

import tensorflow as tf

t = tf.constant(0.525883794, shape=[2,5,5,3], dtype=tf.float32)
tf.raw_ops.L2Loss(t=t)
