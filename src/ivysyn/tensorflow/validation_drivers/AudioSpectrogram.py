# AudioSpectrogramOp

import tensorflow as tf

window_size = 8
stride = 1
magnitude_squared = False
input = tf.constant(-101, shape=[8,1], dtype=tf.float32)
tf.raw_ops.AudioSpectrogram(input=input, window_size=window_size, stride=stride, magnitude_squared=magnitude_squared)
