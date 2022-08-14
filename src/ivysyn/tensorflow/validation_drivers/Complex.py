# BinaryOp

import tensorflow as tf

real = tf.constant(1.41421354, shape=[], dtype=tf.float32)
imag = tf.constant(-1.41421354, shape=[], dtype=tf.float32)
tf.raw_ops.Complex(real=real, imag=imag)
