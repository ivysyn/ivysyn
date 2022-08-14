# QuantizeAndDequantizeV2Op

import tensorflow as tf

signed_input = True
num_bits = 8
range_given = False
round_mode = "HALF_UP"
narrow_range = False
axis = -1
input = tf.constant(-1, shape=[2,3,4,5], dtype=tf.float32)
input_min = tf.constant(0, shape=[], dtype=tf.float32)
input_max = tf.constant(0, shape=[], dtype=tf.float32)
tf.raw_ops.QuantizeAndDequantizeV2(input=input, input_min=input_min, input_max=input_max, signed_input=signed_input, num_bits=num_bits, range_given=range_given, round_mode=round_mode, narrow_range=narrow_range, axis=axis)
