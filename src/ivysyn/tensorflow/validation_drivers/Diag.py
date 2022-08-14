# DiagOp

import tensorflow as tf

diagonal = tf.constant([0.545883179,0.423654795,0.64589411], shape=[3], dtype=tf.float32)
tf.raw_ops.Diag(diagonal=diagonal)
