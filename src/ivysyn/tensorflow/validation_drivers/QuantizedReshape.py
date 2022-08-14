# QuantizedReshapeOp

import tensorflow as tf

tensor = tf.constant(0, shape=[10,20], dtype=tf.quint8)
shape = tf.constant([5,10,4], shape=[3], dtype=tf.int32)
input_min = tf.constant(-10, shape=[1], dtype=tf.float32)
input_max = tf.constant(20, shape=[1], dtype=tf.float32)
tf.raw_ops.QuantizedReshape(tensor=tensor, shape=shape, input_min=input_min, input_max=input_max)
