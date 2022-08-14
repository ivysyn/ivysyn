# MatrixDiagOp

import tensorflow as tf

diagonal = tf.constant(1, shape=[1, 5], dtype=tf.float32)
tf.raw_ops.MatrixDiag(diagonal=diagonal)
