# PopulationCountOp

import tensorflow as tf

x = tf.constant(0, shape=[23], dtype=tf.int8)
tf.raw_ops.PopulationCount(x=x)
