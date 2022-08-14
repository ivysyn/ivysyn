# RandomPoissonOp

import tensorflow as tf

seed = 87654321
seed2 = 0
dtype = tf.float32
shape = tf.constant([5,3], shape=[2], dtype=tf.int32)
rate = tf.constant(1.3, shape=[1], dtype=tf.float32)
tf.raw_ops.RandomPoissonV2(shape=shape, rate=rate, seed=seed, seed2=seed2, dtype=dtype)
