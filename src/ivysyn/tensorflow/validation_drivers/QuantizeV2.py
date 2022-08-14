# QuantizeV2Op

import tensorflow as tf

T = tf.qint8
mode = "MIN_COMBINED"
round_mode = "HALF_AWAY_FROM_ZERO"
narrow_range = False
axis = -1
ensure_minimum_range = 0.01
input = tf.constant(123, shape=[3,4,6,1], dtype=tf.float32)
min_range = tf.constant(-128, shape=[], dtype=tf.float32)
max_range = tf.constant(127, shape=[], dtype=tf.float32)
tf.raw_ops.QuantizeV2(input=input, min_range=min_range, max_range=max_range, T=T, mode=mode, round_mode=round_mode, narrow_range=narrow_range, axis=axis, ensure_minimum_range=ensure_minimum_range)
