# MatrixDiagPartOp

import tensorflow as tf

input = tf.constant(1, shape=[1, 2], dtype=tf.complex128)
tf.raw_ops.MatrixDiagPart(input=input)
