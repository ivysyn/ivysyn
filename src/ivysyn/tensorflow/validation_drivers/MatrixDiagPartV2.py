# MatrixDiagPartOp

import tensorflow as tf

input = tf.constant(1, shape=[1,2], dtype=tf.complex128)
k = tf.constant(0, shape=[], dtype=tf.int32)
padding_value = tf.constant(1, shape=[], dtype=tf.complex128)
tf.raw_ops.MatrixDiagPartV2(input=input, k=k, padding_value=padding_value)
