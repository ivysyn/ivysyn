# SelectV2Op

import tensorflow as tf

condition = tf.constant(True, shape=[5], dtype=tf.bool)
t = tf.constant("A", shape=[5], dtype=tf.string)
e = tf.constant("a", shape=[5], dtype=tf.string)
tf.raw_ops.SelectV2(condition=condition, t=t, e=e)
