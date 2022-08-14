# EinsumOp

import tensorflow as tf

equation = "bij,ijkl->bkl"
inputs = tf.constant(0, shape=[32,5,10], dtype=tf.float32)
tf.raw_ops.Einsum(inputs=inputs, equation=equation)
