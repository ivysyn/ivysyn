# MatrixSetDiagOp

import tensorflow as tf

input = tf.constant([], shape=[0, 0], dtype=tf.complex128)
diagonal = tf.constant([], shape=[0], dtype=tf.complex128)
tf.raw_ops.MatrixSetDiag(input=input, diagonal=diagonal)
