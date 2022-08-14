# QuantizedMulOp

import tensorflow as tf

Toutput = tf.qint32
x = tf.constant(26, shape=[10], dtype=tf.quint8)
y = tf.constant(140, shape=[1], dtype=tf.quint8)
min_x = tf.constant(0, shape=[], dtype=tf.float32)
max_x = tf.constant(10, shape=[], dtype=tf.float32)
min_y = tf.constant(-100, shape=[], dtype=tf.float32)
max_y = tf.constant(100, shape=[], dtype=tf.float32)
tf.raw_ops.QuantizedMul(x=x, y=y, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, Toutput=Toutput)
