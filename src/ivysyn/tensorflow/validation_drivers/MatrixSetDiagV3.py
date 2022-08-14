# MatrixSetDiagOp

import tensorflow as tf

align = "RIGHT_LEFT"
input = tf.constant([], shape=[0,0], dtype=tf.complex128)
diagonal = tf.constant([], shape=[0], dtype=tf.complex128)
k = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.MatrixSetDiagV3(input=input, diagonal=diagonal, k=k, align=align)
