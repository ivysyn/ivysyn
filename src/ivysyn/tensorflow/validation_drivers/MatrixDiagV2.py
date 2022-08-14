# MatrixDiagOp

import tensorflow as tf

diagonal = tf.constant(1, shape=[1,5], dtype=tf.float32)
k = tf.constant(0, shape=[], dtype=tf.int32)
num_rows = tf.constant(-1, shape=[], dtype=tf.int32)
num_cols = tf.constant(-1, shape=[], dtype=tf.int32)
padding_value = tf.constant(0, shape=[], dtype=tf.float32)
tf.raw_ops.MatrixDiagV2(diagonal=diagonal, k=k, num_rows=num_rows, num_cols=num_cols, padding_value=padding_value)
