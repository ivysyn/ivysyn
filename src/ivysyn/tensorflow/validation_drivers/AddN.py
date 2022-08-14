# AddNOp

import tensorflow as tf

inputs = [tf.constant(1.52940166, shape=[], dtype=tf.float32)]
tf.raw_ops.AddN(inputs=inputs)
