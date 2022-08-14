# GetKeyCounterOp

import tensorflow as tf

seed = tf.constant([-2204013331526194987,0], shape=[2], dtype=tf.int64)
tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
