# GuaranteeConstOp

import tensorflow as tf

input = tf.constant(1, shape=[], dtype=tf.float32)
tf.raw_ops.GuaranteeConst(input=input)
