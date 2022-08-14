# CheckNumericsOp

import tensorflow as tf

message = "check-numerics"
tensor = tf.constant([-2,3,-3], shape=[3], dtype=tf.float64)
tf.raw_ops.CheckNumerics(tensor=tensor, message=message)
