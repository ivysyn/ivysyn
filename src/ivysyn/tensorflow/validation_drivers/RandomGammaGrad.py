# BinaryOp

import tensorflow as tf

alpha = tf.constant(1.41421354, shape=[], dtype=tf.float32)
sample = tf.constant(-1.41421354, shape=[], dtype=tf.float32)
tf.raw_ops.RandomGammaGrad(alpha=alpha, sample=sample)
