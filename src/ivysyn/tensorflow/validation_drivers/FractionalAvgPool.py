# FractionalAvgPoolOp

import tensorflow as tf

pooling_ratio = [1, 1.41421354, 1.73205078, 1]
pseudo_random = True
overlapping = True
deterministic = True
seed = 87654321
seed2 = 341261005
value = tf.constant(.90737866941191554, shape=[1,7,13,1], dtype=tf.float64)
tf.raw_ops.FractionalAvgPool(value=value, pooling_ratio=pooling_ratio, pseudo_random=pseudo_random, overlapping=overlapping, deterministic=deterministic, seed=seed, seed2=seed2)
