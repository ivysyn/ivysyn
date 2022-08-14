# QuantizedConcatOp

import tensorflow as tf

concat_dim = tf.constant(0, shape=[], dtype=tf.int32)
values = [tf.constant(1, shape=[2, 2, 3], dtype=tf.quint8),
          tf.constant(133, shape=[2, 2, 3], dtype=tf.quint8)]
input_mins = [tf.constant(0, shape=[], dtype=tf.float32),
              tf.constant(0, shape=[], dtype=tf.float32)]
input_maxes = [tf.constant(255, shape=[], dtype=tf.float32), tf.constant(
    25, shape=[], dtype=tf.float32)]
tf.raw_ops.QuantizedConcat(concat_dim=concat_dim, values=values,
                           input_mins=input_mins, input_maxes=input_maxes)
