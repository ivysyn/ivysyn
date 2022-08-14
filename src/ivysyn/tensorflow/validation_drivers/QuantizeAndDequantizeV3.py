# QuantizeAndDequantizeV3Op

import tensorflow as tf

signed_input = True
range_given = False
narrow_range = False
axis = -1
input = tf.constant(-3.5, shape=[1], dtype=tf.float32)
input_min = tf.constant(0, shape=[], dtype=tf.float32)
input_max = tf.constant(0, shape=[], dtype=tf.float32)
num_bits = tf.constant(8, shape=[], dtype=tf.int32)
tf.raw_ops.QuantizeAndDequantizeV3(input=input, input_min=input_min, input_max=input_max, num_bits=num_bits, signed_input=signed_input, range_given=range_given, narrow_range=narrow_range, axis=axis)
