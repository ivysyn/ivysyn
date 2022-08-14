# ExitOp

import tensorflow as tf

data = tf.constant(1.06872988, shape=[3, 10], dtype=tf.float32)
tf.raw_ops.Exit(data=data)
