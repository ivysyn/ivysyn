# RandomShuffleOp

import tensorflow as tf

seed = 87654321
seed2 = 0
value = tf.constant(0, shape=[10], dtype=tf.int64)
tf.raw_ops.RandomShuffle(value=value, seed=seed, seed2=seed2)
