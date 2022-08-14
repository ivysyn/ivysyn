# EncodeWavOp

import tensorflow as tf

audio = tf.constant(0, shape=[4,2], dtype=tf.float32)
sample_rate = tf.constant(44100, shape=[], dtype=tf.int32)
tf.raw_ops.EncodeWav(audio=audio, sample_rate=sample_rate)
