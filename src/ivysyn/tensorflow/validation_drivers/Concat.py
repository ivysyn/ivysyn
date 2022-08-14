# ConcatBaseOp

import tensorflow as tf

concat_dim = tf.constant([1], shape=[1], dtype=tf.int32)
values = [tf.constant(2, shape=[1], dtype=tf.int32),
          tf.constant(2, shape=[1], dtype=tf.int32)]
tf.raw_ops.Concat(concat_dim=concat_dim, values=values)
